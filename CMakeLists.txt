cmake_minimum_required(VERSION 3.1.0)

project(face_landmark_detect)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/headDetec)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/rs2pcd)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/facemesh)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

target_link_libraries(${PROJECT_NAME} 
                        headDetect
                        facemesh
                     )

message(${PROJECT_SOURCE_DIR})


