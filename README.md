# faceReconstruction
人脸建模

运行程序
 cmake ../examples -DUSE_SSE2_INSTRUCTIONS=ON
 
人脸图片68特征点检测并将特侦点写入pts文件
./face_landmark_detection_ex ../share/shape_predictor_68_face_landmarks.dat ../data/images/*.jpg


模型计算
./fit-model

