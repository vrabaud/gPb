project( textons )
find_package( OpenCV REQUIRED )
add_executable( textons textons )
target_link_libraries( textons ${OpenCV_LIBS} )
