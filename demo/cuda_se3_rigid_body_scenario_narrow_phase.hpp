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
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/narrowphase/collision.h>
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

        // maximum and minimum coordinates, for axis aligned bounding box (AABB) porpuses
        float min[3];
        float max[3];
        
        __host__ __device__ void update_aabb(){
            
            for (size_t i = 0; i < 3; i++){
                float curr_min = A[i];
                float curr_max = A[i];

                if (curr_min > B[i]){
                    curr_min = B[i];
                }
                
                if (curr_max < B[i]){
                    curr_max = B[i];
                }

                if (curr_min > C[i]){
                    curr_min = C[i];
                }

                if (curr_max < C[i]){
                    curr_max = C[i];
                }

                min[i] = curr_min;
                max[i] = curr_max;
            }
        }

        Triangle() {}
        __host__ __device__ Triangle(Vec3 vertex_a, Vec3 vertex_b, Vec3 vertex_c):
            A(vertex_a),
            B(vertex_b),
            C(vertex_c) {
                update_aabb();
            }
        
        __host__ __device__ bool aabbOverlap(Triangle other) {
            return  (!(min[0] > other.max[0] || max[0] < other.min[0])) 
                    &&  (!(min[1] > other.max[1] || max[1] < other.min[1])) 
                    &&  (!(min[2] > other.max[2] || max[2] < other.min[2]));

        }
    };



    template <typename Scalar>
    class Mesh {
        using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
        using Transform = Eigen::Transform<Scalar, 3, Eigen::Affine>;

        std::string name_;
        fcl::BVHModel<fcl::OBBRSS<Scalar>> model_;

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

            model_.beginModel();
            std::size_t nTris = visitTriangles(
                scene,
                scene->mRootNode,
                rootTransform,
                [&] (const Vec3& a, const Vec3& b, const Vec3& c) {
                    model_.addTriangle(a, b, c);
                });
            model_.endModel();
            model_.computeLocalAABB();

            for (int i = 0; i < model_.num_tris; i++) {
                host_triangles_.emplace_back(   model_.vertices[model_.tri_indices[i][0]],
                                                model_.vertices[model_.tri_indices[i][1]],
                                                model_.vertices[model_.tri_indices[i][2]]);
            }

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
__device__ __host__ int coplanar_tri_tri(float N[3],float V0[3],float V1[3],float V2[3],
                     float U0[3],float U1[3],float U2[3]);

// some vector macros

    #define THRESHOLD 0.01f

    #define FABS(x) (x>=0?x:-x)        /* implement as is fastest on your machine */

    #define CROSS(dest,v1,v2)                       \
                dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
                dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
                dest[2]=v1[0]*v2[1]-v1[1]*v2[0];



    #define   sVpsV_2( Vr, s1,  V1,s2, V2);\
        {\
    Vr[0] = s1*V1[0] + s2*V2[0];\
    Vr[1] = s1*V1[1] + s2*V2[1];\
    }\

    #define myVpV(g,v2,v1);\
    {\
        g[0] = v2[0]+v1[0];\
        g[1] = v2[1]+v1[1];\
        g[2] = v2[2]+v1[2];\
        }\

    #define myVmV(g,v2,v1);\
    {\
        g[0] = v2[0]-v1[0];\
        g[1] = v2[1]-v1[1];\
        g[2] = v2[2]-v1[2];\
        }\

    // 2D intersection of segment and triangle.
    //  Q = {0,0, -150,000,000}
    // r = (-7, 39, -25) * 10,000,000
    #define seg_collide3( q, r)\
    {\
        p1[0]=SF*P1[0];\
        p1[1]=SF*P1[1];\
        p2[0]=SF*P2[0];\
        p2[1]=SF*P2[1];\
        det1 = p1[0]*q[1]-q[0]*p1[1]; /*det1 = 0 */ \
        det1_sign = det1>=-THRESHOLD? 1:-1;\
        gama1 = (p1[0]*r[1]-r[0]*p1[1])*det1_sign;\
        alpha1 = (r[0]*q[1] - q[0]*r[1])*det1_sign;\
        alpha1_legal = (alpha1>= 0) && (alpha1<=(FABS(det1))  && (FABS(det1) > THRESHOLD));\
        det2 = p2[0]*q[1] - q[0]*p2[1];\
        det2_sign = det2>=-THRESHOLD? 1:-1;\
        alpha2 = (r[0]*q[1] - q[0]*r[1]) *det2_sign;\
        gama2 = (p2[0]*r[1] - r[0]*p2[1]) * det2_sign;\
        alpha2_legal = (alpha2>= 0) && (alpha2<=(FABS(det2)) && (FABS(det2) > THRESHOLD));\
        det3=det2-det1;\
        det3_sign = det3>=-THRESHOLD? 1:-1;\
        gama3=((p2[0]-p1[0])*(r[1]-p1[1]) - (r[0]-p1[0])*(p2[1]-p1[1]))*det3_sign;\
        if (alpha1_legal)\
        {\
            if (alpha2_legal)\
            {\
                if ( ((gama1<= 0) && (gama1>=-(FABS(det1)))) || ((gama2<= 0) && (gama2>=-(FABS(det2)))) || (gama1*gama2< 0)) return 12;\
            }\
            else\
            {\
                if ( ((gama1<= 0) && (gama1>=-(FABS(det1)))) || ((gama3<= 0) && (gama3>=-(FABS(det3)))) || (gama1*gama3< 0)) return 13;\
                }\
        }\
        else\
        if (alpha2_legal)\
        {\
            if ( ((gama2<= 0) && (gama2>=-(FABS(det2)))) || ((gama3<= 0) && (gama3>=-(FABS(det3)))) || (gama2*gama3< 0)) return 23;\
            }\
        return 0;\
        }




    //main procedure

    __device__ __host__ int tr_tri_intersect3D (float C1[3], float P1[3], float P2[3],
            float D1[3], float Q1[3], float Q2[3])
    {
        float  t[3],p1[3], p2[3],r[3],r4[3];
        float beta1, beta2, beta3;
        float gama1, gama2, gama3;
        float det1, det2, det3;
        float det1_sign, det2_sign, det3_sign;
        float dp0, dp1, dp2;
        float dq1,dq2,dq3,dr, dr3;
        float alpha1, alpha2;
        bool alpha1_legal, alpha2_legal;
        float  SF;
        bool beta1_legal, beta2_legal;

        myVmV(r,D1,C1);
        // determinant computation
        dp0 = P1[1]*P2[2]-P2[1]*P1[2]; // 0
        dp1 = P1[0]*P2[2]-P2[0]*P1[2]; // dp1 = -1000
        dp2 = P1[0]*P2[1]-P2[0]*P1[1]; // 0
        dq1 = Q1[0]*dp0 - Q1[1]*dp1 + Q1[2]*dp2; // dq1 = 100,000
        dq2 = Q2[0]*dp0 - Q2[1]*dp1 + Q2[2]*dp2; // 0
        dr  = -r[0]*dp0  + r[1]*dp1  - r[2]*dp2; // dr = 40,000



        beta1 = dr*dq2;  // beta1, beta2 are scaled so that beta_i=beta_i*dq1*dq2 // 0
        beta2 = dr*dq1; // 4,000,000,000
        beta1_legal = (beta2>= 0) && (beta2 <=dq1*dq1) && (dq1 != 0); // true
        beta2_legal = (beta1>= 0) && (beta1 <=dq2*dq2) && (dq2 != 0); // false

        dq3=dq2-dq1; //-100,000
        dr3=+dr-dq1;   // actualy this is -dr3 // -60,000


        if ((FABS(dq1) < THRESHOLD) && ( FABS(dq2) < THRESHOLD))
        {
            if (dr != 0) return 0;  // triangles are on parallel planes
            else
            {						// triangles are on the same plane
                float C2[3],C3[3],D2[3],D3[3], N1[3];
                // We use the coplanar test of Moller which takes the 6 vertices and 2 normals
                //as input.
                myVpV(C2,C1,P1);
                myVpV(C3,C1,P2);
                myVpV(D2,D1,Q1);
                myVpV(D3,D1,Q2);
                CROSS(N1,P1,P2);
                if (coplanar_tri_tri(N1,C1, C2,C3,D1,D2,D3)){
                    return true;
                };
                return false;
            }
        }

        else if (!beta2_legal && !beta1_legal) return 0;// fast reject -- all vertices are on
                                                        // the same side of the triangle plane

        else if (beta2_legal && beta1_legal)    //beta1, beta2
        {
            SF = dq1*dq2;
            // printf("beta1, beta2"))
            sVpsV_2(t,beta2,Q2, (-beta1),Q1);
        }

        else if (beta1_legal && !beta2_legal)   //beta1, beta3
        {
            SF = dq1*dq3; // -10,000,000,000
            beta1 =beta1-beta2;   // all betas are multiplied by a positive SF // -4,000,000
            beta3 =dr3*dq1; // -6,000,000,000
            sVpsV_2(t,(SF-beta3-beta1),Q1,beta3,Q2); //t = -6,000,000 * Q2
        }

        else if (beta2_legal && !beta1_legal) //beta2, beta3
        {
            SF = dq2*dq3;
            beta2 =beta1-beta2;   // all betas are multiplied by a positive SF
            beta3 =dr3*dq2;
            sVpsV_2(t,(SF-beta3),Q1,(beta3-beta2),Q2);
            Q1=Q2;
            beta1=beta2;
        }
        sVpsV_2(r4,SF,r,beta1,Q1); // r4 {0, D1[1] - C1[1], undefined \ 0}  * 10,000,000 
        seg_collide3(t,r4);  // calculates the 2D intersection // t = Q2 * -6,000,000 // r4 = (D1 - C1) * 10,000,000
        return 0;
    }

        /* this edge to edge test is based on Franlin Antonio's gem:
    "Faster Line Segment Intersection", in Graphics Gems III,
    pp. 199-202 */
    #define FABS(x) (x>=0?x:-x)        /* implement as is fastest on your machine */

    #define EDGE_EDGE_TEST(V0,U0,U1)                      \
    Bx=U0[i0]-U1[i0];                                   \
    By=U0[i1]-U1[i1];                                   \
    Cx=V0[i0]-U0[i0];                                   \
    Cy=V0[i1]-U0[i1];                                   \
    f=Ay*Bx-Ax*By;                                      \
    d=By*Cx-Bx*Cy;                                      \
    if((f>THRESHOLD && d>= THRESHOLD && d<=f) || (f< -THRESHOLD && d<= -THRESHOLD && d>=f))  \
    {                                                   \
        e=Ax*Cy-Ay*Cx;                                    \
        if(f>THRESHOLD)                                           \
        {                                                 \
        if(e>= THRESHOLD && e<=f) return 1;                      \
        }                                                 \
        else                                              \
        {                                                 \
        if(e<= -THRESHOLD && e>=f) return 1;                      \
        }                                                 \
    }

    #define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
    {                                              \
    float Ax,Ay,Bx,By,Cx,Cy,e,d,f;               \
    Ax=V1[i0]-V0[i0];                            \
    Ay=V1[i1]-V0[i1];                            \
    /* test edge U0,U1 against V0,V1 */          \
    EDGE_EDGE_TEST(V0,U0,U1);                    \
    /* test edge U1,U2 against V0,V1 */          \
    EDGE_EDGE_TEST(V0,U1,U2);                    \
    /* test edge U2,U1 against V0,V1 */          \
    EDGE_EDGE_TEST(V0,U2,U0);                    \
    }

    #define POINT_IN_TRI(V0,U0,U1,U2)           \
    {                                           \
    float a,b,c,d0,d1,d2;                     \
    /* is T1 completly inside T2? */          \
    /* check if V0 is inside tri(U0,U1,U2) */ \
    a=U1[i1]-U0[i1];                          \
    b=-(U1[i0]-U0[i0]);                       \
    c=-a*U0[i0]-b*U0[i1];                     \
    d0=a*V0[i0]+b*V0[i1]+c;                   \
                                                \
    a=U2[i1]-U1[i1];                          \
    b=-(U2[i0]-U1[i0]);                       \
    c=-a*U1[i0]-b*U1[i1];                     \
    d1=a*V0[i0]+b*V0[i1]+c;                   \
                                                \
    a=U0[i1]-U2[i1];                          \
    b=-(U0[i0]-U2[i0]);                       \
    c=-a*U2[i0]-b*U2[i1];                     \
    d2=a*V0[i0]+b*V0[i1]+c;                   \
    if(d0*d1>THRESHOLD)                             \
    {                                         \
        if(d0*d2>THRESHOLD) return 1;                 \
    }                                         \
    }

    //This procedure testing for intersection between coplanar triangles is taken
    // from Tomas Moller's
    //"A Fast Triangle-Triangle Intersection Test",Journal of Graphics Tools, 2(2), 1997
    __device__ __host__ int coplanar_tri_tri(float N[3],float V0[3],float V1[3],float V2[3],
                        float U0[3],float U1[3],float U2[3])
    {
    float A[3];
    short i0,i1;
    /* first project onto an axis-aligned plane, that maximizes the area */
    /* of the triangles, compute indices: i0,i1. */
    A[0]=FABS(N[0]);
    A[1]=FABS(N[1]);
    A[2]=FABS(N[2]);
    if(A[0]>A[1])
    {
        if(A[0]>A[2])
        {
            i0=1;      /* A[0] is greatest */
            i1=2;
        }
        else
        {
            i0=0;      /* A[2] is greatest */
            i1=1;
        }
    }
    else   /* A[0]<=A[1] */
    {
        if(A[2]>A[1])
        {
            i0=0;      /* A[2] is greatest */
            i1=1;
        }
        else
        {
            i0=0;      /* A[1] is greatest */
            i1=2;
        }
        }

        /* test all edges of triangle 1 against the edges of triangle 2 */
        EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2);
        EDGE_AGAINST_TRI_EDGES(V1,V2,U0,U1,U2);
        EDGE_AGAINST_TRI_EDGES(V2,V0,U0,U1,U2);

        /* finally, test if tri1 is totally contained in tri2 or vice versa */
        POINT_IN_TRI(V0,U0,U1,U2);
        POINT_IN_TRI(U0,V0,V1,V2);

        return 0;
    }

    // void detect_collision_all_robot_host(
    //         impl::Triangle<float> *obstacles, size_t obs_size,
    //         impl::Triangle<float> *robot, size_t rob_size,
    //         bool *collisions, Eigen::Transform<float, 3, Eigen::Isometry> tf){

    //     float threshold = 0.0001f;


    //     for (size_t idx = 0; idx < obs_size ; idx++){
    //         impl::Triangle<float> obs = obstacles[idx];
    //         // Vec3 obs_vec1 = obs.B - obs.A;
    //         // Vec3 obs_vec2 = obs.C - obs.A;


    //         // Vec3 obs_norm = obs_vec1.cross(obs_vec2);

    //         // // scalar that satisfies obs_norm * X + obs_d = 0dot
    //         // float obs_d = -1 * obs_norm.dot(obs.A);

    //         // // case where vertices of a 'triangle' are colinear- ignore collision
    //         // // todo, do line intersection test with triangle
    //         // if (fabsf(obs_norm[0]) < threshold && fabsf(obs_norm[1]) < threshold && fabsf(obs_norm[2]) < threshold){
    //         //     collisions[idx] = false;

    //         //     return;
    //         // }


    //         float obs_center_vertex[] = {obs.A[0], obs.A[1], obs.A[2]};
    //         float obs_edge_B[] =      {obs.B[0] - obs.A[0], obs.B[1] - obs.A[1], obs.B[2] - obs.A[2]};
    //         float obs_edge_C[] =      {obs.C[0] - obs.A[0], obs.C[1] - obs.A[1], obs.C[2] - obs.A[2]};
    //         //test for intersection against all robot triangles
    //         ////////////////////////////////////////////////////////////////////////
    //         bool has_collision = false;
    //         for (int i = 0; i < rob_size; i++){
    //             impl::Triangle<float> pre_trans_rob = robot[i];
    //             impl::Triangle<float> rob(  tf * pre_trans_rob.A,
    //                                         tf * pre_trans_rob.B,
    //                                         tf * pre_trans_rob.C);
    //                 int collision_idx = obs_idx * rob_size + rob_idx;

    //             // preliminary check to see if the triangles axis-aligned bounding boxes intersect
    //             if (!rob.aabbOverlap(obs)){
    //                 continue;
    //             }

    //             float rob_center_vertex[] = {rob.A[0], rob.A[1], rob.A[2]};
    //             float rob_edge_B[] =      {rob.B[0] - rob.A[0], rob.B[1] - rob.A[1], rob.B[2] - rob.A[2]};
    //             float rob_edge_C[] =      {rob.C[0] - rob.A[0], rob.C[1] - rob.A[1], rob.C[2] - rob.A[2]};

    //             if (tr_tri_intersect3D      (obs_center_vertex, obs_edge_B, obs_edge_C,
    //                                          rob_center_vertex, rob_edge_B, rob_edge_C)){
    //                 has_collision=true;
    //                 collisions[idx] = has_collision;

    //                 printf("Robot Triangle %d: \n{(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)} \nintersects triangle\n{(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)}\n with rob AABB \n{(%f, %f, %f), (%f, %f, %f)}\n and obs AABB \n{(%f, %f, %f), (%f, %f, %f)}\n", i,
    //                     rob_center_vertex[0], rob_center_vertex[1], rob_center_vertex[2],
    //                     rob_edge_B[0], rob_edge_B[1], rob_edge_B[2],
    //                     rob_edge_C[0], rob_edge_C[1], rob_edge_C[2],
    //                     obs_center_vertex[0], obs_center_vertex[1], obs_center_vertex[2],
    //                     obs_edge_B[0], obs_edge_B[1], obs_edge_B[2],
    //                     obs_edge_C[0], obs_edge_C[1], obs_edge_C[2],
    //                     rob.min[0], rob.min[1], rob.min[2],
    //                     rob.max[0], rob.max[1], rob.max[2],
    //                     obs.min[0], obs.min[1], obs.min[2],
    //                     obs.max[0], obs.max[1], obs.max[2]);
    //                 break;
    //             }
    //         }
    //         collisions[idx] = has_collision;
    //     }

    // }





    // TODO: change indexing so every thread in a warp has the same robot triangle
    // single triangle triangle collision detection
    __global__  void detect_collision_one_robot_tri(
            impl::Triangle<float> *obstacles, size_t obs_size,
            impl::Triangle<float> *robot, size_t rob_size,
            bool *collisions, Eigen::Transform<float, 3, Eigen::Isometry> tf){

        int global_index = threadIdx.x + blockIdx.x * blockDim.x;
        int obs_idx = global_index / rob_size;
        int rob_idx = global_index % rob_size;
        // edge case where thread doesn't matter
        if (obs_idx >= obs_size){
            return;
        }
        
        if (rob_idx >= rob_size){
            return;
        }

        // ORDER IS IMPORTANT
        int collision_idx = obs_idx * rob_size + rob_idx;

        impl::Triangle<float> obs = obstacles[obs_idx];


        float obs_center_vertex[] = {obs.A[0], obs.A[1], obs.A[2]};
        float obs_edge_B[] =      {obs.B[0] - obs.A[0], obs.B[1] - obs.A[1], obs.B[2] - obs.A[2]};
        float obs_edge_C[] =      {obs.C[0] - obs.A[0], obs.C[1] - obs.A[1], obs.C[2] - obs.A[2]};

        float rob_center_vertex[3];
        float rob_edge_B[3];
        float rob_edge_C[3];
        //test for intersection against all robot triangles
        ////////////////////////////////////////////////////////////////////////
        bool has_collision = false;
        impl::Triangle<float> pre_trans_rob = robot[rob_idx];
        impl::Triangle<float> rob(  tf * pre_trans_rob.A,
                                    tf * pre_trans_rob.B,
                                    tf * pre_trans_rob.C);


        // preliminary check to see if the triangles axis-aligned bounding boxes intersect
        collisions[collision_idx] = rob.aabbOverlap(obs);
    }

    // one environment triangle vs all robot triangles
    // void detect_collision_all_robot(
    __global__  void detect_collision_all_robot(
            impl::Triangle<float> *obstacles, size_t obs_size,
            impl::Triangle<float> *robot, size_t rob_size,
            bool *collisions, Eigen::Transform<float, 3, Eigen::Isometry> tf){

        // want some tolerance when comparing to 0 to account for float errors
        float threshold = 0.0001f;

        int obs_idx = threadIdx.x + blockIdx.x * blockDim.x;

        // edge case where thread doesn't matter
        if (obs_idx >= obs_size){
            return;
        }
        // if (idx == 0){
            // printf("There are %d obstacle triangles", obs_size );
            // for (int i = 0; i < obs_size; i++){
            //     printf("Triangle %d: {(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)} \n ", i,
            //                         obstacles[i].A[0], obstacles[i].A[1], obstacles[i].A[2],
            //                         obstacles[i].B[0], obstacles[i].B[1], obstacles[i].B[2],
            //                         obstacles[i].C[0], obstacles[i].C[1], obstacles[i].C[2]);
            // }
        //     printf("There are %d robot triangles", rob_size );
        //     // for (int i = 0; i < rob_size; i++){
        //     //     printf("Triangle %d: {(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)} \n ", i,
        //     //                         robot[i].A[0], robot[i].A[1], robot[i].A[2],
        //     //                         robot[i].B[0], robot[i].B[1], robot[i].B[2],
        //     //                         robot[i].C[0], robot[i].C[1], robot[i].C[2]);
        //     // }
        // }
        impl::Triangle<float> obs = obstacles[obs_idx];

        // test for intersection against all robot triangles' aabb's
        ////////////////////////////////////////////////////////////////////////
        for (int i = 0; i < rob_size; i++){
            impl::Triangle<float> pre_trans_rob = robot[i];
            impl::Triangle<float> rob(  tf * pre_trans_rob.A,
                                        tf * pre_trans_rob.B,
                                        tf * pre_trans_rob.C);

            int collision_idx = obs_idx * rob_size + i;


            // preliminary check to see if the triangles axis-aligned bounding boxes intersect
            collisions[collision_idx] = rob.aabbOverlap(obs);
        }
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
        bool *m_host_collisions;
        bool *m_d_collisions;
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


            size_t block_size = 256;

            cudaDeviceSynchronize();
            // size_t numBlocks = num_env_triangles / block_size +1;
            // detect_collision_all_robot<<< numBlocks, 256>>>(  environment_->d_triangles_, num_env_triangles,
            //                                         (*robot_)[0].d_triangles_, num_rob_triangles,
            //                                         m_d_collisions, tf);
            
            size_t numBlocks = ((num_env_triangles * num_rob_triangles) / block_size) +1;
            detect_collision_one_robot_tri<<< numBlocks, 256>>>(  environment_->d_triangles_, num_env_triangles,
                                                    (*robot_)[0].d_triangles_, num_rob_triangles,
                                                    m_d_collisions, tf);
            cudaDeviceSynchronize();

            cudaMemcpy(m_host_collisions, m_d_collisions, num_env_triangles * num_rob_triangles * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            bool isValid = true;
            for (int i = 0; i <num_env_triangles * num_rob_triangles; i ++){
                if (m_host_collisions[i]){

                    int obs_idx = i / num_rob_triangles;
                    int rob_idx = i % num_rob_triangles;
                    impl::Triangle<float> obs = environment_->host_triangles_[obs_idx];

                    float obs_center_vertex[] = {obs.A[0], obs.A[1], obs.A[2]};
                    float obs_edge_B[] =      {obs.B[0] - obs.A[0], obs.B[1] - obs.A[1], obs.B[2] - obs.A[2]};
                    float obs_edge_C[] =      {obs.C[0] - obs.A[0], obs.C[1] - obs.A[1], obs.C[2] - obs.A[2]};

                    impl::Triangle<float> pre_trans_rob = (*robot_)[0].host_triangles_[rob_idx];
                    impl::Triangle<float> rob(  tf * pre_trans_rob.A,
                                                tf * pre_trans_rob.B,
                                                tf * pre_trans_rob.C);

                    float rob_center_vertex[] = {rob.A[0], rob.A[1], rob.A[2]};
                    float rob_edge_B[] =      {rob.B[0] - rob.A[0], rob.B[1] - rob.A[1], rob.B[2] - rob.A[2]};
                    float rob_edge_C[] =      {rob.C[0] - rob.A[0], rob.C[1] - rob.A[1], rob.C[2] - rob.A[2]};
                    
                    if (tr_tri_intersect3D      (obs_center_vertex, obs_edge_B, obs_edge_C,
                                                 rob_center_vertex, rob_edge_B, rob_edge_C)){
                        return false;
                    }
                }
            }

            return isValid;

            // TODO - test on Easy ~ .024 seconds, cubicles ~ 0.5 seconds, and Home ~ 2.5 seconds
            // TODO - write to a single global flag instead of an array of collisions
            // TODO - check the collisions in a loop instead of using cudaDeviceSynchronize, continuing
            //        whenever we've determined there's a collision
        }





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

            int num_collisions =  environment_->host_triangles_.size() * (*robot_)[0].host_triangles_.size();
            m_host_collisions = new bool[num_collisions];
            cudaMalloc((void **) &m_d_collisions, sizeof(bool) * num_collisions);

            MPT_LOG(DEBUG) << "Volume min: " << min.transpose();
            MPT_LOG(DEBUG) << "Volume max: " << max.transpose();
            // MPT_LOG(DEBUG) << "state type: " << log::type_name<State>()
        }

        ~SE3RigidBodyScenario(){
            // if (m_d_collisions)
            // delete m_host_collisions;
            cudaFree(m_d_collisions);
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

