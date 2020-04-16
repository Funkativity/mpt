// https://web.stanford.edu/class/cs277/resources/papers/Moller1997b.pdf
// ideas for further work - precompute all robot mesh norms
#include "MeshParser.h"
#include <chrono>

__device__ float dot(float3 a, float3 b){
    return ( a.x * b.x 
            +a.y * b.y
            +a.z * b.z);
}

__device__ float3 vecMinus(float3 a, float3 b){
    return  {a.x - b.x, 
             a.y - b.y,
             a.z - b.z};
}

__device__ float3 cross(float3 a, float3 b){
    return  {a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x};
}

// returns the t1 value as seen in equation (4) in the cited paper
__device__ float getParam(float p0, float p1, float d0, float d1){
    return (p0 + (p1 - p0) * (d0) / (d0 - d1));
}

__global__ void detect_collision(   
        Triangle *obstacles, size_t obs_size,
        Triangle *robot, size_t rob_size,
        bool *collisions){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // edge case where thread doesn't matter
    if (idx >= obs_size){
        return;
    }
    
    Triangle obs = obstacles[idx];

    // calculate normal for our obstacle triangle
    ////////////////////////////////////////////////////////////////////////
    
    float3 obs_vec1 = vecMinus(obs.B, obs.A);
    float3 obs_vec2 = vecMinus(obs.C, obs.A);

    
    float3 obs_norm = cross(obs_vec1, obs_vec2);

    // scalar that satisfies obs_norm * X + obs_d = 0
    float obs_d = -1 * dot(obs_norm, obs.A); 


    //test for intersection against all robot triangles
    ////////////////////////////////////////////////////////////////////////
    Triangle rob;
    bool has_collision = false;
    for (int i = 0; i < rob_size; i++){
        rob = robot[i];
        float3 obs_planar_distances;

        // note: x, y, and z represent which the distances
        // from the triangle to the obstacle plane, not coordinates
        obs_planar_distances.x = dot(obs_norm, rob.A) + obs_d;
        obs_planar_distances.y = dot(obs_norm, rob.B) + obs_d;
        obs_planar_distances.z = dot(obs_norm, rob.C) + obs_d;

        // coplanar case
        //TODO add rosetta code to my citations
        if (abs(obs_planar_distances.x + obs_planar_distances.y + obs_planar_distances.z) < 0.0001f) {
            //TODO, also refactor code so this can appear later 

            //TODO - project vertices onto a flat plane
            bool (*chkEdge)(TriPoint &, TriPoint &, TriPoint &, double) = NULL;
            
            has_collision = true;
            //For edge E of trangle 1,
            for(int i=0; i<3; i++)
            {
                int j=(i+1)%3;
        
                //Check all points of trangle 2 lay on the external side of the edge E. If
                //they do, the triangles do not collide.
                if (chkEdge(t1[i], t1[j], t2[0], eps) &&
                    chkEdge(t1[i], t1[j], t2[1], eps) &&
                    chkEdge(t1[i], t1[j], t2[2], eps)){


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
                    if (chkEdge(t2[i], t2[j], t1[0], eps) &&
                        chkEdge(t2[i], t2[j], t1[1], eps) &&
                        chkEdge(t2[i], t2[j], t1[2], eps)){

                        has_collision = false;
                        break;

                    }

                }
        
            if(has_collision){
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
        float3 rob_vec1 = vecMinus(obs.B, obs.A);
        float3 rob_vec2 = vecMinus(obs.C, obs.A);
        float3 rob_norm = cross(obs_vec1, obs_vec2);

        // scalar that satisfies obs_norm * X + obs_d = 0
        float rob_d = -1 * dot(obs_norm, obs.A); 

        float3 rob_planar_distances;
        rob_planar_distances.x = dot(rob_norm, obs.A) + rob_d;
        rob_planar_distances.y = dot(rob_norm, obs.B) + rob_d;
        rob_planar_distances.z = dot(rob_norm, obs.C) + rob_d;

        if ((rob_planar_distances.x > 0 && rob_planar_distances.y > 0 && rob_planar_distances.z > 0)
                || (rob_planar_distances.x < 0 && rob_planar_distances.y < 0 && rob_planar_distances.z < 0)){
            continue;
        }
        ///////////////////////////////////////////////////////////////////////////////////
        float3 direction = cross(rob_norm, obs_norm);

        // get points of obs intersecting line and corresponding planar distance
        float obs_intersect1, obs_intersect2;
        float obs_distance1, obs_distance2;
        if (rob_planar_distances.x > 0){
            if (rob_planar_distances.y > 0){
                obs_intersect1 = dot(direction, obs.A);
                obs_intersect2 = dot(direction, obs.B);
                obs_distance1 = rob_planar_distances.x;
                obs_distance2 = rob_planar_distances.y;
            } else {
                obs_intersect1 = dot(direction, obs.A);
                obs_intersect2 = dot(direction, obs.C);
                obs_distance1 = rob_planar_distances.x;
                obs_distance2 = rob_planar_distances.z;
            }
        } else {
            obs_intersect1 = dot(direction, obs.B);
            obs_intersect2 = dot(direction, obs.C);
            obs_distance1 = rob_planar_distances.y;
            obs_distance2 = rob_planar_distances.z;
        }
        
        // get points of rob intersecting line
        float rob_intersect1, rob_intersect2;
        float rob_distance1, rob_distance2;
        if (obs_planar_distances.x > 0){
            if (obs_planar_distances.y > 0){
                rob_intersect1 = dot(direction, rob.A);
                rob_intersect2 = dot(direction, rob.B);
                rob_distance1 = obs_planar_distances.x;
                rob_distance2 = obs_planar_distances.y;
            } else {
                rob_intersect1 = dot(direction, rob.A);
                rob_intersect2 = dot(direction, rob.C);
                rob_distance1 = obs_planar_distances.x;
                rob_distance2 = obs_planar_distances.z;
            }
        } else {
            rob_intersect1 = dot(direction, rob.B);
            rob_intersect2 = dot(direction, rob.C);
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
            has_collision = true;
            break;
        }

    }

    collisions[idx] = has_collision;
}

float dotc(float3 a, float3 b){
    return ( (a.x * b.x) 
            +(a.y * b.y)
            +(a.z * b.z));
}

float3 vecMinusc(float3 a, float3 b){
    return  {a.x - b.x, 
             a.y - b.y,
             a.z - b.z};
}

float3 crossc(float3 a, float3 b){
    return  {a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x};
}

// returns the t1 value as seen in equation (4) in the cited paper
float getParamc(float p0, float p1, float d0, float d1){
    return (p0 + (p1 - p0) * (d0) / (d0 - d1));
}

void detectCollisionCPU(std::vector<Triangle> &robot, std::vector<Triangle> &obstacles, bool *collisions){
    for (int idx = 0; idx < obstacles.size(); idx++){
        Triangle obs = obstacles[idx];

        // calculate normal for our obstacle triangle
        ////////////////////////////////////////////////////////////////////////
        
        float3 obs_vec1 = vecMinusc(obs.B, obs.A);
        float3 obs_vec2 = vecMinusc(obs.C, obs.A);
    
        
        float3 obs_norm = crossc(obs_vec1, obs_vec2);
    
        // scalar that satisfies obs_norm * X + obs_d = 0
        float obs_d = -1 * dotc(obs_norm, obs.A); 
    
    
        //test for intersection against all robot triangles
        ////////////////////////////////////////////////////////////////////////
        Triangle rob;
        bool has_collision = false;
        for (int i = 0; i < robot.size(); i++){
            rob = robot[i];
            float3 obs_planar_distances;
    
            // note: x, y, and z represent which the distances
            // from the triangle to the obstacle plane, not coordinates
            obs_planar_distances.x = dotc(obs_norm, rob.A) + obs_d;
            obs_planar_distances.y = dotc(obs_norm, rob.B) + obs_d;
            obs_planar_distances.z = dotc(obs_norm, rob.C) + obs_d;
    
            // coplanar case
            if (abs(obs_planar_distances.x + obs_planar_distances.y + obs_planar_distances.z) < 0.0001f) {
                //TODO, also refactor code so this can appear later
            }
    
            // may want to change 0 to some small threshhold above 0 to allow for coplanar case
            if ((obs_planar_distances.x > 0 && obs_planar_distances.y > 0 && obs_planar_distances.z > 0)
                    || (obs_planar_distances.x < 0 && obs_planar_distances.y < 0 && obs_planar_distances.z < 0)){
                continue;
            }
    
            ///////////////////////////////////////////////////////////////////////////////////
            // calculate the projection of the obstacle triangle against the robot triangle now
            float3 rob_vec1 = vecMinusc(obs.B, obs.A);
            float3 rob_vec2 = vecMinusc(obs.C, obs.A);
            float3 rob_norm = crossc(obs_vec1, obs_vec2);
    
            // scalar that satisfies obs_norm * X + obs_d = 0
            float rob_d = -1 * dotc(obs_norm, obs.A); 
    
            float3 rob_planar_distances;
            rob_planar_distances.x = dotc(rob_norm, obs.A) + rob_d;
            rob_planar_distances.y = dotc(rob_norm, obs.B) + rob_d;
            rob_planar_distances.z = dotc(rob_norm, obs.C) + rob_d;
    
            if ((rob_planar_distances.x > 0 && rob_planar_distances.y > 0 && rob_planar_distances.z > 0)
                    || (rob_planar_distances.x < 0 && rob_planar_distances.y < 0 && rob_planar_distances.z < 0)){
                continue;
            }
            ///////////////////////////////////////////////////////////////////////////////////
            float3 direction = crossc(rob_norm, obs_norm);
    
            // get points of obs intersecting line and corresponding planar distance
            float obs_intersect1, obs_intersect2;
            float obs_distance1, obs_distance2;
            if (rob_planar_distances.x > 0){
                if (rob_planar_distances.y > 0){
                    obs_intersect1 = dotc(direction, obs.A);
                    obs_intersect2 = dotc(direction, obs.B);
                    obs_distance1 = rob_planar_distances.x;
                    obs_distance2 = rob_planar_distances.y;
                } else {
                    obs_intersect1 = dotc(direction, obs.A);
                    obs_intersect2 = dotc(direction, obs.C);
                    obs_distance1 = rob_planar_distances.x;
                    obs_distance2 = rob_planar_distances.z;
                }
            } else {
                obs_intersect1 = dotc(direction, obs.B);
                obs_intersect2 = dotc(direction, obs.C);
                obs_distance1 = rob_planar_distances.y;
                obs_distance2 = rob_planar_distances.z;
            }
            
            // get points of rob intersecting line
            float rob_intersect1, rob_intersect2;
            float rob_distance1, rob_distance2;
            if (obs_planar_distances.x > 0){
                if (obs_planar_distances.y > 0){
                    rob_intersect1 = dotc(direction, rob.A);
                    rob_intersect2 = dotc(direction, rob.B);
                    rob_distance1 = obs_planar_distances.x;
                    rob_distance2 = obs_planar_distances.y;
                } else {
                    rob_intersect1 = dotc(direction, rob.A);
                    rob_intersect2 = dotc(direction, rob.C);
                    rob_distance1 = obs_planar_distances.x;
                    rob_distance2 = obs_planar_distances.z;
                }
            } else {
                rob_intersect1 = dotc(direction, rob.B);
                rob_intersect2 = dotc(direction, rob.C);
                rob_distance1 = obs_planar_distances.y;
                rob_distance2 = obs_planar_distances.z;
            }
    
            // should probably refactor these above if statements so that this is a part of it
            // get parameters such that intersection = obs_paramx * D 
            float obs_param1 = getParamc(    obs_intersect1, obs_intersect2, 
                                            obs_distance1, obs_distance2);
    
            float obs_param2 = getParamc(    obs_intersect2, obs_intersect1, 
                                            obs_distance2, obs_distance1);
            
            float rob_param1 = getParamc(    rob_intersect1, rob_intersect2, 
                                            rob_distance1, rob_distance2);
        
            float rob_param2 = getParamc(    rob_intersect2, rob_intersect1, 
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
                has_collision = true;
                break;
            }
    
        }
        
        collisions[idx] = has_collision;
    }
}

int main(){
    // load meshes
    // since we only want binary information about the presence of a collision
    // we can pass all of our meshes together as one concatenated array of triangles.
    // future work can involve calculating indices into this array of triangles by 
    // also copying over an array of offsets into the device, allowing separation of individual 
    // meshes.

    // note, this does not detect whether the robot is entirely inside the mesh
    // this problem can be ignored by choosing a step size smaller than the minimum radius
    // of the robot when implementing this as a part of RRT, since then we guarantee that at 
    // least one point on the path will be in strictly intersecting an obstacle if there is 
    // any collision at all.
    cudaDeviceSynchronize();
    for (int i = 10; i < 10000001 ; i *= 10){
        std::cout << "\nTest with " << i << " obstacle triangles" <<std::endl;
        std::vector <Triangle> obstacles;
        std::vector <Triangle> robot;
        bool *collisions_GPU;
        bool *collisions_CPU;
        // bool ok = loadTrianglesEncapsulated("meshes/cube.obj", obstacles);
        bool ok = loadTrianglesEncapsulated("meshes/cube.obj", robot);
        addRandomTriangles(obstacles, i, 100, 100, 100, .01 );
        if (!ok){
            std::cout << "Error, mesh could not be read" <<std::endl;
            exit(-1);
        }

        collisions_GPU = new bool[obstacles.size()];
        collisions_CPU = new bool[obstacles.size()];

        // load meshes into GPU, timed
        Triangle *d_obstacles, *d_robot;
        bool *d_collisions;

        auto start = std::chrono::high_resolution_clock::now();
        cudaMalloc((void **) &d_obstacles, sizeof(Triangle) * obstacles.size());
        cudaMalloc((void **) &d_collisions, sizeof(bool) * obstacles.size());
        cudaMalloc((void **) &d_robot, sizeof(Triangle) * robot.size());

        cudaMemcpy(d_obstacles, &obstacles[0], obstacles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
        cudaMemcpy(d_robot, &robot[0], robot.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clcollisions_GPUock::now();
        std::chrono::duration<double> elapsed = end  - start;
        std::cout << "GPU Memory writing took " << elapsed.count() << " seconds " << std::endl;
        // execute kernel, timed

        int numBlocks = obstacles.size() / 256 + 1;
        int num_obs_triangles = obstacles.size();
        int num_rob_triangles = robot.size();
        start = std::chrono::high_resolution_clock::now();
        detect_collision<<< numBlocks, 256>>>(d_obstacles, num_obs_triangles, d_robot, num_rob_triangles, d_collisions);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        // load result, timed
        elapsed = end  - start;
        std::cout << "GPU Execution took " << elapsed.count() << " seconds " << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(collisions_GPU, d_collisions, obstacles.size() * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(d_obstacles);
        cudaFree(d_robot);
        cudaFree(d_collisions);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        elapsed = end  - start;
        std::cout << "GPU Memory reading took " << elapsed.count() << " seconds " << std::endl;
        free (collisions_GPU);


        //CPU benchmarking
        start = std::chrono::high_resolution_clock::now();
        detectCollisionCPU(robot, obstacles, collisions_CPU);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end  - start;
        std::cout << "CPU execution took " << elapsed.count() << " seconds " << std::endl;
        int mismatches = 0;
        int GPU_falses = 0;
        int CPU_falses = 0;
        for (int j = 0; j < obstacles.size(); j++){
            if (collisions_CPU[j]){
                CPU_falses++;
            }
            if (collisions_GPU[j]){
                GPU_falses++;
            }
            if (collisions_CPU[j] != collisions_GPU[j]){
                mismatches++;
            }
        }
        std::cout << "Mismatches: " << mismatches << std::endl;
        std::cout << "GPU falses: " << GPU_falses << std::endl;
        std::cout << "CPU falses: " << CPU_falses << std::endl;
        free (collisions_CPU);
    }
}