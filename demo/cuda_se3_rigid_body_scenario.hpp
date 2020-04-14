// Software License Agreement (BSD-3-Clause)
//
// Copyright 2018 The University of North Carolina at Chapel Hill
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

//! @author Jeff Ichnowski

#pragma once

#include <Eigen/Dense>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
// #include <fcl/geometry/bvh/BVH_model.h>
// #include <fcl/narrowphase/collision.h>
#include <functional>
#include <memory>
#include <mpt/discrete_motion_validator.hpp>
#include <mpt/goal_state.hpp>
#include <mpt/log.hpp>
#include <mpt/se3_space.hpp>
#include <mpt/uniform_sampler.hpp>
#include <nigh/kdtree_batch.hpp>

#include <cuda.h>

#define BATCH_SIZE 32
#define NN_TYPE KDTreeBatch
#define SCALAR_TYPE float
namespace mpt_demo::impl {
    static constexpr std::intmax_t SO3_WEIGHT = 50;
    using Vec3 = Eigen::Matrix<float, 3, 1>;

    // silence the warnings "taking address of packed member 'a1' of
    // class or structure 'aiMatrix4x4t<float>' may result in an
    // unaligned pointer value" We use static_asserts to ensure
    // unaligned structures will not happen.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"

    template <typename Scalar>
    auto mapToEigen(const aiMatrix4x4t<Scalar>& m) {
        using EigenType = const Eigen::Matrix<Scalar, 4, 4, Eigen::RowMajor>;
        static_assert(sizeof(EigenType) == sizeof(m));
        return Eigen::Map<EigenType>(&m.a1);
    }

    template <typename Scalar>
    auto mapToEigen(const aiVector3t<Scalar>& v) {
        using EigenType = const Eigen::Matrix<Scalar, 3, 1>;
        static_assert(sizeof(EigenType) == sizeof(v));
        return Eigen::Map<const EigenType>(&v.x);
    }
#pragma GCC diagnostic pop

    template <typename Scalar, typename Fn>
    std::size_t visitVertices(
        const aiScene* scene, const aiNode *node,
        Eigen::Transform<Scalar, 3, Eigen::Affine> transform,
        Fn&& visitor)
    {
        std::size_t count = 0;
        transform *= mapToEigen(node->mTransformation).template cast<Scalar>();
        for (unsigned i=0 ; i < node->mNumMeshes ; ++i) {
            const aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
            count += mesh->mNumVertices;
            for (unsigned j=0 ; j < mesh->mNumVertices ; ++j)
                visitor(transform * mapToEigen(mesh->mVertices[j]).template cast<Scalar>());
        }
        for (unsigned i=0 ; i < node->mNumChildren ; ++i)
            count += visitVertices(scene, node->mChildren[i], transform, std::forward<Fn>(visitor));
        return count;
    }

    template <typename Scalar, typename Fn>
    static std::size_t visitTriangles(
        const aiScene *scene, const aiNode *node,
        Eigen::Transform<Scalar, 3, Eigen::Affine> transform,
        Fn&& visitor)
    {
        std::size_t count = 0;

        transform *= mapToEigen(node->mTransformation).template cast<Scalar>();
        for (unsigned i=0 ; i<node->mNumMeshes ; ++i) {
            const aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
            for (unsigned j=0 ; j<mesh->mNumFaces ; ++j) {
                const aiFace& face = mesh->mFaces[j];
                if (face.mNumIndices < 3)
                    continue;

                // Support trangular decomposition by fanning out
                // around vertex 0.  The indexing follows as:
                //
                //   0---1   0 1 2
                //  /|\ /    0 2 3
                // 4-3-2     0 3 4
                //
                Vec3 v0 = transform * mapToEigen(mesh->mVertices[face.mIndices[0]]).template cast<Scalar>();
                Vec3 v1 = transform * mapToEigen(mesh->mVertices[face.mIndices[1]]).template cast<Scalar>();
                for (unsigned k=2 ; k<face.mNumIndices ; ++k) {
                    Vec3 v2 = transform * mapToEigen(mesh->mVertices[face.mIndices[k]]).template cast<Scalar>();
                    visitor(v0, v1, v2);
                    v1 = v2;
                }
                count += face.mNumIndices - 2;
            }
        }
        for (unsigned i=0 ; i<node->mNumChildren ; ++i)
            count += visitTriangles(scene, node->mChildren[i], transform, std::forward<Fn>(visitor));

        return count;
    }



    // Triangle class encapsulates coordinates of vertices, instead of storing
    // vertex indices and then looking them up in an array.
    // Having local copies of vertex coordinates is less space effecient, but
    // reduces number of accesses to standard memory. It also creates more total
    // work to do, since now transformations need to be done multiple times for
    // each vertex, but since the transformations will be parallelized,  we believe
    // this trade off will be negligible in terms of time saved accessing memory.
    template <typename Scalar>
    class Triangle {
        using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

    public:
        // coordinates of triangle vertices
        Vec3 A;
        Vec3 B;
        Vec3 C;

        Triangle() {}

        __host__ __device__ Triangle(Vec3 vertex_a, Vec3 vertex_b, Vec3 vertex_c):
            A(vertex_a),
            B(vertex_b),
            C(vertex_c) {}


    };



    template <typename Scalar>
    class Mesh {
        using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
        using Transform = Eigen::Transform<Scalar, 3, Eigen::Affine>;

        std::string name_;

    public:
        std::vector<Triangle<Scalar>> host_triangles_; // RAM copy of triangles
        Triangle<Scalar> *d_triangles_; //pointer to GPU copy of triangles for a mesh

        Mesh(const std::string& name, bool shiftToCenter)
            : name_(name)
        {
            Assimp::Importer importer;
            // these options are the same as from OMPL app to strive
            // for parity
            static constexpr auto readOpts =
                aiProcess_Triangulate | aiProcess_JoinIdenticalVertices |
                aiProcess_SortByPType | aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes;

            const aiScene *scene = importer.ReadFile(name, readOpts);
            if (scene == nullptr)
                throw std::invalid_argument("could not load mesh file '" + name + "'");

            if (!scene->HasMeshes())
                throw std::invalid_argument("mesh file '" + name + "' does not contain meshes");

            Vec3 center = Vec3::Zero();
            std::size_t nVertices = visitVertices(
                scene,
                scene->mRootNode,
                Transform::Identity(),
                [&] (const Vec3& v) { center += v; });
            center /= nVertices;

            Transform rootTransform = Transform::Identity();

            if (shiftToCenter)
                rootTransform *= Eigen::Translation<Scalar, 3>(-center);

            host_triangles_.reserve(nVertices); // could be an unhelpful optimization, but worth a try!
            std::size_t nTris = visitTriangles(
                scene,
                scene->mRootNode,
                rootTransform,
                [&] (const Vec3& a, const Vec3& b, const Vec3& c) {
                    host_triangles_.emplace_back(a, b, c);
                });

            cudaMalloc((void **) &d_triangles_, sizeof(Triangle<Scalar>) * host_triangles_.size());

            cudaMemcpy(d_triangles_, &host_triangles_[0], host_triangles_.size() * sizeof(Triangle<float>), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize(); // consider not having cudaDeviceSynchronize here, instead maybe call it at the start of valid.

            MPT_LOG(INFO) << "Loaded mesh '" << name << "' (" << nVertices << " vertices, " << nTris
                          << " triangles, center=" << center << ")";
        }


    };

    template <auto>
    class member_function;

    template <typename T, typename R, R T::* fn>
    class member_function<fn> {
        T& obj_;
    public:
        member_function(T& obj) : obj_(obj) {}
        template <typename ... Args>
        decltype(auto) operator() (Args&& ... args) { return (obj_.*fn)(std::forward<Args>(args)...); }
        template <typename ... Args>
        decltype(auto) operator() (Args&& ... args) const { return (obj_.*fn)(std::forward<Args>(args)...); }
    };
}



namespace mpt_demo {
    using namespace unc::robotics;
    using Vec3 = Eigen::Matrix<float, 3, 1>;


    // CUDA helper functions
    //////////////////////////////////////////////////////////////////////////////////


    __device__ inline float det_2d(float2 &p1, float2 &p2, float2 &p3) {
        return  p1.x*(p2.y-p3.y)
                +p2.x*(p3.y-p1.y)
                +p3.x*(p1.y-p2.y);
    }

    // swaps the vertices of a 2d triangle so that they're in the order
    // needed for the 2d_triangle_intersect test
    __device__  void CheckTriWinding(float2 &p1, float2 &p2, float2 &p3){
        float detTri = det_2d(p1, p2, p3);
        if(detTri < 0.0)
        {
            float2 a = p3;
            p3 = p2;
            p2 = a;
        }
    }



    __device__ inline float getTriAreaSquared(float2 a, float2 b, float2 c){
        float a_len_squared =   (b.x - c.x) * (b.x - c.x) + 
                                (b.y - c.y) * (b.y - c.y); 

        float b_len_squared =   (a.x - c.x) * (a.x - c.x) + 
                                (a.y - c.y) * (a.y - c.y); 

        float c_len_squared =   (a.x - b.x) * (a.x - b.x) + 
                                (a.y - b.y) * (a.y - b.y); 
        
        float semi_perimeter = (a_len_squared + b_len_squared + c_len_squared) / 2.0f;

        return semi_perimeter * (semi_perimeter - a_len_squared)
                              * (semi_perimeter - b_len_squared)
                              * (semi_perimeter - c_len_squared);
    }



    __device__ inline bool edge_intersection(float2 &p1, float2 &p2, float2 &p3, double eps){
        return det_2d(p1, p2, p3) < eps;
    }

    // returns the t1 value as seen in equation (4) in the cited paper
    __device__ float getParam(float p0, float p1, float d0, float d1){
        return (p0 + (p1 - p0) * (d0) / (d0 - d1));
    }

    __global__ void detect_collision(
            impl::Triangle<float> *obstacles, size_t obs_size,
            impl::Triangle<float> *robot, size_t rob_size,
            bool *collisions, Eigen::Transform<float, 3, Eigen::Isometry> tf){

        // want some tolerance when comparing to 0 to account for float errors
        float threshold = 0.00001f;

        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // edge case where thread doesn't matter
        if (idx >= obs_size){
            return;
        }
        if (idx == 0){
            // printf("There are %d obstacle triangles", obs_size );
            // for (int i = 0; i < obs_size; i++){
            //     printf("Triangle %d: {(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)} \n ", i,
            //                         obstacles[i].A[0], obstacles[i].A[1], obstacles[i].A[2],
            //                         obstacles[i].B[0], obstacles[i].B[1], obstacles[i].B[2],
            //                         obstacles[i].C[0], obstacles[i].C[1], obstacles[i].C[2]);
            // }
            printf("There are %d robot triangles", rob_size );
            // for (int i = 0; i < rob_size; i++){
            //     printf("Triangle %d: {(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)} \n ", i,
            //                         robot[i].A[0], robot[i].A[1], robot[i].A[2],
            //                         robot[i].B[0], robot[i].B[1], robot[i].B[2],
            //                         robot[i].C[0], robot[i].C[1], robot[i].C[2]);
            // }
        }
        impl::Triangle<float> obs = obstacles[idx];

        // calculate normal for our obstacle triangle
        ////////////////////////////////////////////////////////////////////////

        Vec3 obs_vec1 = obs.B - obs.A;
        Vec3 obs_vec2 = obs.C - obs.A;


        Vec3 obs_norm = obs_vec1.cross(obs_vec2);

        // scalar that satisfies obs_norm * X + obs_d = 0dot
        float obs_d = -1 * obs_norm.dot(obs.A);


        //test for intersection against all robot triangles
        ////////////////////////////////////////////////////////////////////////
        bool has_collision = false;
        for (int i = 0; i < rob_size; i++){
            impl::Triangle<float> pre_trans_rob = robot[i];
            impl::Triangle<float> rob(  tf * pre_trans_rob.A, 
                                        tf * pre_trans_rob.B,
                                        tf * pre_trans_rob.C);
            // if (idx == 0){
            //     printf("Triangle %d: {(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)} \n ", i,
            //         robot[i].A[0], robot[i].A[1], robot[i].A[2],
            //         robot[i].B[0], robot[i].B[1], robot[i].B[2],
            //         robot[i].C[0], robot[i].C[1], robot[i].C[2]);
            
            // }
            float3 obs_planar_distances;

            // note: x, y, and z represent which the distances
            // from the triangle to the obstacle plane, not coordinates
            obs_planar_distances.x = obs_norm.dot(rob.A) + obs_d;
            obs_planar_distances.y = obs_norm.dot(rob.B) + obs_d;
            obs_planar_distances.z = obs_norm.dot(rob.C) + obs_d;

            // coplanar case
            //TODO add rosetta code to my citations
            if (abs(obs_planar_distances.x + obs_planar_distances.y + obs_planar_distances.z) < threshold) {
                //TODO, also refactor code so this can appear later 
                float2 t1[3], t2[3];
                has_collision = true;

                // case project triangles onto axis aligned plane that gives them greatest area
                // this prevents accidentally projecting triangles into lines
                // use the semi perimeter method of calculating triangle areas
                // since we are only comparing positive values, we avoid taking 
                // square roots for performance
                float2 xy_a =make_float2(obs.A[0], obs.A[1]);
                float2 xy_b =make_float2(obs.B[0], obs.B[1]);
                float2 xy_c =make_float2(obs.C[0], obs.C[1]);

                float2 xz_a =make_float2(obs.A[0], obs.A[2]);
                float2 xz_b =make_float2(obs.B[0], obs.B[2]);
                float2 xz_c =make_float2(obs.C[0], obs.C[2]);

                float2 yz_a =make_float2(obs.A[1], obs.A[2]);
                float2 yz_b =make_float2(obs.B[1], obs.B[2]);
                float2 yz_c =make_float2(obs.C[1], obs.C[2]);
                
                float xy_area_squared_squared = getTriAreaSquared(xy_a, xy_b, xy_c);
                float xz_area_squared_squared = getTriAreaSquared(xz_a, xz_b, xz_c);
                float yz_area_squared_squared = getTriAreaSquared(yz_a, yz_b, yz_c);

                // TODO- refactor this into a function
                if (xy_area_squared_squared >= xz_area_squared_squared &&
                    xy_area_squared_squared >= yz_area_squared_squared) {
                    t1[0] = xy_a;
                    t1[1] = xy_b;
                    t1[2] = xy_c;

                    t2[0] = make_float2(rob.A[0], rob.A[1]);
                    t2[1] = make_float2(rob.B[0], rob.B[1]);
                    t2[2] = make_float2(rob.C[0], rob.C[1]);
                } else if (xz_area_squared_squared >= yz_area_squared_squared) {
                    t1[0] = xz_a;
                    t1[1] = xz_b;
                    t1[2] = xz_c;

                    t2[0] = make_float2(rob.A[0], rob.A[2]);
                    t2[1] = make_float2(rob.B[0], rob.B[2]);
                    t2[2] = make_float2(rob.C[0], rob.C[2]);
                } else {
                    t1[0] = yz_a;
                    t1[1] = yz_b;
                    t1[2] = yz_c;

                    t2[0] = make_float2(rob.A[1], rob.A[2]);
                    t2[1] = make_float2(rob.B[1], rob.B[2]);
                    t2[2] = make_float2(rob.C[1], rob.C[2]);
                }
                         
                //Trangles must be expressed anti-clockwise
                CheckTriWinding(t1[0], t1[1], t1[2]);
                CheckTriWinding(t2[0], t2[1], t2[2]);

                //For edge E of trangle 1,
                for(int i=0; i<3; i++)
                {
                    int j=(i+1)%3;
            
                    //Check all points of trangle 2 lay on the external side of the edge E. If
                    //they do, the triangles do not collide.
                    if (edge_intersection(t1[i], t1[j], t2[0], threshold) &&
                        edge_intersection(t1[i], t1[j], t2[1], threshold) &&
                        edge_intersection(t1[i], t1[j], t2[2], threshold)){


                        has_collision = false;
                        break;
                    }
                }
                
                if (!has_collision)
                    //For edge E of trangle 2,
                    for(int i=0; i<3; i++)
                    {
                        int j=(i+1)%3;
                
                        //Check all points of trangle 1 lay on the external side of the edge E. If
                        //they do, the triangles do not collide.
                        if (edge_intersection(t2[i], t2[j], t1[0], threshold) &&
                            edge_intersection(t2[i], t2[j], t1[1], threshold) &&
                            edge_intersection(t2[i], t2[j], t1[2], threshold)){

                            has_collision = false;
                            break;

                        }

                    }
            
                if(has_collision){
                    printf("Obstacle triangle %d is in collision with robot triangle %d,", idx, i);
                    printf("rob triangle: ");
                    break;
                }
                else {
                    continue;
                }
            }


            // may want to change 0 to some small threshhold above 0 to allow for coplanar case
            if ((obs_planar_distances.x > 0 && obs_planar_distances.y > 0 && obs_planar_distances.z > 0)
                    || (obs_planar_distances.x < 0 && obs_planar_distances.y < 0 && obs_planar_distances.z < 0)){
                continue;
            }

            ///////////////////////////////////////////////////////////////////////////////////
            // calculate the projection of the obstacle triangle against the robot triangle now
            Vec3 rob_vec1 = obs.B - obs.A;
            Vec3 rob_vec2 = obs.C - obs.A;
            Vec3 rob_norm = rob_vec1.cross(rob_vec2);

            // scalar that satisfies obs_norm * X + obs_d = 0
            float rob_d = -1 * rob_norm.dot(rob.A);

            float3 rob_planar_distances;
            rob_planar_distances.x = rob_norm.dot(obs.A) + rob_d;
            rob_planar_distances.y = rob_norm.dot(obs.B) + rob_d;
            rob_planar_distances.z = rob_norm.dot(obs.C) + rob_d;

            if ((rob_planar_distances.x > threshold && rob_planar_distances.y > threshold && rob_planar_distances.z > threshold)
                    || (rob_planar_distances.x < -threshold && rob_planar_distances.y < -threshold && rob_planar_distances.z < -threshold)){
                continue;
            }
            ///////////////////////////////////////////////////////////////////////////////////

            // this is the direction of the line created by the intersection of the planes
            // of both triangles
            Vec3 direction = rob_norm.cross(obs_norm);

            // get points of obs intersecting line and corresponding planar distance
            float obs_intersect1, obs_intersect2;
            float obs_distance1, obs_distance2;
            if (rob_planar_distances.x > 0){
                if (rob_planar_distances.y > 0){
                    obs_intersect1 = direction.dot(obs.A);
                    obs_intersect2 = direction.dot(obs.B);
                    obs_distance1 = rob_planar_distances.x;
                    obs_distance2 = rob_planar_distances.y;
                } else {
                    obs_intersect1 = direction.dot(obs.A);
                    obs_intersect2 = direction.dot(obs.C);
                    obs_distance1 = rob_planar_distances.x;
                    obs_distance2 = rob_planar_distances.z;
                }
            } else {
                obs_intersect1 = direction.dot(obs.B);
                obs_intersect2 = direction.dot(obs.C);
                obs_distance1 = rob_planar_distances.y;
                obs_distance2 = rob_planar_distances.z;
            }

            // get points of rob intersecting line
            float rob_intersect1, rob_intersect2;
            float rob_distance1, rob_distance2;
            if (obs_planar_distances.x > 0){
                if (obs_planar_distances.y > 0){
                    rob_intersect1 = direction.dot(rob.A);
                    rob_intersect2 = direction.dot(rob.B);
                    rob_distance1 = obs_planar_distances.x;
                    rob_distance2 = obs_planar_distances.y;
                } else {
                    rob_intersect1 = direction.dot(rob.A);
                    rob_intersect2 = direction.dot(rob.C);
                    rob_distance1 = obs_planar_distances.x;
                    rob_distance2 = obs_planar_distances.z;
                }
            } else {
                rob_intersect1 = direction.dot(rob.B);
                rob_intersect2 = direction.dot(rob.C);
                rob_distance1 = obs_planar_distances.y;
                rob_distance2 = obs_planar_distances.z;
            }

            // should probably refactor these above if statements so that this is a part of it
            // get parameters such that intersection = obs_paramx * D
            float obs_param1 = getParam(    obs_intersect1, obs_intersect2,
                                            obs_distance1, obs_distance2);

            float obs_param2 = getParam(    obs_intersect2, obs_intersect1,
                                            obs_distance2, obs_distance1);

            float rob_param1 = getParam(    rob_intersect1, rob_intersect2,
                                            rob_distance1, rob_distance2);

            float rob_param2 = getParam(    rob_intersect2, rob_intersect1,
                                            rob_distance2, rob_distance1);

            // swap so that 1 is smaller
            if (obs_param1 > obs_param2) {
                float tmp = obs_param2;
                obs_param2 = obs_param1;
                obs_param1 = tmp;
            }
            if (rob_param1 > rob_param2) {
                float tmp = rob_param2;
                rob_param2 = rob_param1;
                rob_param1 = tmp;
            }

            if ( (obs_param2 < rob_param1) || obs_param1 > rob_param1) {
                continue; // no collision
            } else {
                printf("Obstacle triangle %d is in collision with robot triangle %d,", idx, i);
                printf("rob triangle: ");
                has_collision = true;
                break;
            }

        }
        if (has_collision){
            printf("Obstacle triangle %d is in collision with robot triangle %d,", idx, i);
            printf("rob triangle: ");
        }
        collisions[idx] = has_collision;
    }


    template <typename Scalar, int nParts = 1, bool selfCollision = false>
    class SE3RigidBodyScenario {

        static_assert(nParts == 1, "only single body motions are supported currently (TODO: add multibody)");

    public:

        using Space = mpt::SE3Space<Scalar, impl::SO3_WEIGHT>; // weight SO(3) by 50
        using Bounds = std::tuple<mpt::Unbounded, mpt::BoxBounds<Scalar, 3>>;
        using State = typename Space::Type;
        using Distance = typename Space::Distance;
        using Goal = mpt::GoalState<Space>;
        using TravelTime = Scalar;

        // TODO: remove explicit Nearest and use default
        using Nearest = unc::robotics::nigh::KDTreeBatch<8>;

    private:
        using Config = typename Space::Type;
        // fcl::Transform3<Scalar> is an alias for
        // Eigen::Transform<Scalar, 3, Eigen::Isometry> though in a
        // previous version the last template parameter was
        // Eigen::AffineCompact.  Isometry seems like a better option,
        // so it seems unlikely to change, but regardless, we use
        // fcl's alias for it instead of directly using Eigen's type.
        using Transform = Eigen::Transform<Scalar, 3, Eigen::Isometry>;

        // The meshes are immutable and can be rather large.  Since
        // scenarios are copied to each thread, we use a shared
        // pointer to avoid copying the environment and robot meshes.
        std::shared_ptr<impl::Mesh<Scalar>> environment_;
        std::shared_ptr<std::vector<impl::Mesh<Scalar>>> robot_;

        Space space_;
        Bounds bounds_;

        static constexpr Distance goalRadius = 1e-6;
        Goal goal_;

        static Transform stateToTransform(const Config& q) {
            return Eigen::Translation<Scalar, 3>(std::get<1>(q))
                * Eigen::Quaternion(std::get<0>(q));
        }

    public:

        // the collision detection function
        // is called in prrt.hpp or pprm.hpp or whatever algorithm this is compiled to use
        bool valid(const Config& q) const {

            int num_env_triangles = environment_->host_triangles_.size();

            if (robot_->size() > 1){
                throw new std::runtime_error("Only single robots supported");
            }
            size_t num_rob_triangles = (*robot_)[0].host_triangles_.size();

            
            Transform tf = stateToTransform(q);

            // array of booleans, d_collisions[i] = true -> environment_.triangle[i] is
            bool host_collisions[num_env_triangles];
            bool *d_collisions;
            cudaMalloc((void **) &d_collisions, sizeof(bool) * num_env_triangles);

            size_t block_size = 256;
            size_t numBlocks = num_env_triangles / block_size +1;


            // for (int i = 0; i < num_rob_triangles; i++){

            //     printf("Triangle %d: {(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)} \n ", i,
            //                         (*robot_)[0].host_triangles_[i].A[0], (*robot_)[0].host_triangles_[i].A[1], (*robot_)[0].host_triangles_[i].A[2],
            //                         (*robot_)[0].host_triangles_[i].B[0], (*robot_)[0].host_triangles_[i].B[1], (*robot_)[0].host_triangles_[i].B[2],
            //                         (*robot_)[0].host_triangles_[i].C[0], (*robot_)[0].host_triangles_[i].C[1], (*robot_)[0].host_triangles_[i].C[2]);
            // }

            cudaDeviceSynchronize();
            detect_collision<<< numBlocks, 256>>>(  environment_->d_triangles_, num_env_triangles,
                                                    (*robot_)[0].d_triangles_, num_rob_triangles,
                                                    d_collisions, tf);
            cudaDeviceSynchronize();

            cudaMemcpy(host_collisions, d_collisions, num_env_triangles * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            bool isValid = true;
            for (int i = 0; i <num_env_triangles; i ++){
                isValid = isValid && !host_collisions[i];
                if (host_collisions[i]){
                    // int zero = 0;
                    std::cout << "env_triangle " << i <<"collides ";
                }
            }
            std::cout << isValid << std::endl;

            return isValid;


            // TODO - write to a single global flag instead of an array of collisions
            // TODO - have each CUDA thread check a single triangle triangle, instead of
            //        all triangles in a robot
            // TODO - check the collisions in a loop instead of using cudaDeviceSynchronize, continuing
            //        whenever we've determined there's a collision0\
            // TODO - initialize vector of bools in construction of scenario, to avoid excessive calls
            //        to cudaMalloc and CUDA free
            // TODO - precompute norms for environment / robot, transform as necessary.
        }

        // // TODO make this use use some cuda algorithm
        // std::vector<bool> validBatch(const std::vector<Config> qs) const {
        //     std::vector<Transform> tfs[robot_->size()];
        //     std::vector<bool> collisions[robot_->size()];

        //     for (const auto& q : qs) {
        //         for (const auto& robot : *robot_) {
        //             for (size_t j = 0)
        //             // consider doing the "state to transform" within the gpu
        //             // could calculate the transform on cpu for one robot point, send a batch, calculate the next transform, send another batch, etc
        //             // ^ that's a good idea, i like it.
        //             Transform tf = stateToTransform(q);
        //         }
        //     }

        //     return collisions;
        // }


    private:
        using Validator = impl::member_function<&SE3RigidBodyScenario::valid>;

    public:
        const mpt::DiscreteMotionValidator<Space, Validator> link_;

        template <typename Min, typename Max>
        SE3RigidBodyScenario(
            const std::string& envMesh,
            const std::vector<std::string>& robotMeshes,
            const Config& goal,
            const Eigen::MatrixBase<Min>& min,
            const Eigen::MatrixBase<Max>& max,
            Scalar checkResolution)
            : environment_(std::make_shared<impl::Mesh<Scalar>>(envMesh, false))
            , robot_(std::make_shared<std::vector<impl::Mesh<Scalar>>>())
            , bounds_(mpt::Unbounded{}, mpt::BoxBounds<Scalar, 3>(min, max)) // environment_.minBounds(), environment_.maxBounds())),
            , goal_(goalRadius, goal)
            // , link_(space_, environment_->extents()*checkResolution, Validator(*this))
            , link_(space_, ((max - min).norm() + Scalar(impl::SO3_WEIGHT*M_PI/2))*checkResolution, Validator(*this))
        {
            robot_->reserve(robotMeshes.size());
            for (const std::string& mesh : robotMeshes) {
                robot_->emplace_back(mesh, true);
            }
            MPT_LOG(DEBUG) << "Volume min: " << min.transpose();
            MPT_LOG(DEBUG) << "Volume max: " << max.transpose();
        }

        const Space& space() const {
            return space_;
        }

        const Bounds& bounds() const {
            return bounds_;
        }

        const Goal& goal() const {
            return goal_;
        }

        bool link(const Config& a, const Config& b) const {
            return link_(a, b);
        }

        // TODO: this shouldn't be necessary
        TravelTime travelTime(Distance dist, bool) const { return dist; }
    };
}

