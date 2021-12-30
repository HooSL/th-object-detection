# th-object-detection

설치하는 방법
먼저 아나콘다가 설치
https://www.anaconda.com/products/individual

다 설치를 했다면, 비쥬얼 스튜디오 코드에서 cmd 터미널 창띄우기
(비쥬얼 스튜디오 코드 상단바에 terminal에서  new terminal 누릅니다.)

+ 옆에 화살표를 눌러 command prompt를 누르고 왼쪽에 (base)가 나오는지 확인
파이썬 가상환경을 만들기

해당 터미널에 conda create -n tensorflow pip python=3.9 실행

-n 뒤에 tensorflow는 가상환경 이름이므로 다른걸로 해도 됨

conda activate tensorflow 실행해서 가상환경으로 설정

입력하면 (base)가 (tensorflow)로 변경

pip install --ignore-installed --upgrade tensorflow==2.5.0 실행해서 텐서플로우 설치

설치가 완료되면 설치 확인
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))" 실행

실행해서
2020-06-22 19:20:32.614181: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-06-22 19:20:32.620571: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2020-06-22 19:20:35.027232: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-06-22 19:20:35.060549: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties:
pciBusID: 0000:02:00.0 name: GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 8.00GiB deviceMemoryBandwidth: 238.66GiB/s
2020-06-22 19:20:35.074967: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-06-22 19:20:35.084458: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cublas64_10.dll'; dlerror: cublas64_10.dll not found
2020-06-22 19:20:35.094112: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2020-06-22 19:20:35.103571: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2020-06-22 19:20:35.113102: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
2020-06-22 19:20:35.123242: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusparse64_10.dll'; dlerror: cusparse64_10.dll not found
2020-06-22 19:20:35.140987: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-06-22 19:20:35.146285: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1598] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-06-22 19:20:35.162173: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-06-22 19:20:35.178588: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x15140db6390 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-22 19:20:35.185082: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-06-22 19:20:35.191117: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-22 19:20:35.196815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]
tf.Tensor(1620.5817, shape=(), dtype=float32)

비슷하게 나오면 성공

tensorflow를 설치했으므로 이제 tensorflow object detection API를 설치할 차례
https://github.com/tensorflow/models







