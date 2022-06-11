# NTUST Edge AI (110-02) 期末專題報告_M11015Q10_章聰誠
## 1. 作品名稱：【智慧交通】─交通號誌辨識
## 2. 摘要說明
智慧交通已是近年來熱門的話題，而在交通號誌辨識同樣也達成智慧交通的其中一環，因此嘗試自己實作做為學習，詳細文檔可參考以下網址，本篇將以說明操作為主。
文檔說明連結:https://hackmd.io/@cczhang708/SyLQ9Nmbq![image](https://user-images.githubusercontent.com/101262636/173178164-545d03f2-7180-4d0d-82d4-c968a935254f.png)

辨識模型:yolov4-tiny

流程:
1. 在colab訓練辨識號誌之yolov4-tiny模型
2. 將模型轉換為IR格式(OpenVINO推論用的格式)
3. 利用 OpenVINO 進行 demo
### 如何執行
***1.在colab訓練辨識號誌之yolov4-tiny模型***
利用github內"yolov4-tiny_training_test_final.ipynb"於colab進行yolov4-tiny訓練，只要照內部文件說明執行就可得到訓練好的權重檔(*.weight)
或利用連結:https://colab.research.google.com/drive/1FyQpNAJwUzaMdO3X_ZUNHTnmtlGKw2WQ?usp=sharing
![image](https://user-images.githubusercontent.com/101262636/173178475-4c34cbf7-b8d4-4201-97bc-9a17d19c481f.png)
以上連結與github內"yolov4-tiny_training_test_final.ipynb"相同

資料集圖片共有498張，batch size為112，訓練時間至少2.5小時。
資料集連結:https://drive.google.com/file/d/17yLjZQi1P15L9IouFQhRSkNvDgOXZty4/view?usp=sharing
yolov4-tiny_training_test_final.ipynb以內附此訊息，執行時會自動下載相關資料。
其中"my_yolov4-tiny-custom_final.weights"為我以訓練好的模型。

如想自己訓練自被資料集可參考:
https://omnixri.blogspot.com/2021/05/google-colabyolov4-tiny.html

***2.將模型轉換為IR格式***

由於本人使用的Openvino版本為2021.4，並沒有直接將yolov4轉為IR格式，因此先用公開套件(以下連結，非官方)將模型轉.pb，再使用OpeVINO官方轉為IR格式，進而完成推理。
參考: https://github.com/TNTWEN/OpenVINO-YOLOV4

我已將我訓練好的模型轉IR檔:frozen_darknet_yolov4_model.bin、frozen_darknet_yolov4_model.xml，放在github供大家參考
推論時需要*.bin及*.xml，且要放在相同資料夾內

環境:
1.	OS:WIN10
2.	Python version:3.6
3.	Tensorflow 1.12	#一定要用1.X版，2.X有改指令，所以會出問題(除非作者有更新，但我用的時候還是舊的)

步驟
1.	從上方網址下載壓縮檔
![](https://i.imgur.com/2KJGq0W.png)

2.	解壓縮(我是到桌面)


---

之後在CMD下執行(你可以用anoconda)

本人是使用anoconda prompt 執行
我有附 conda 的環境包(model_trans_environment.yml)，你可以使用以下指令安裝:

```
 conda env create -f model_trans_environment.yml
```
3.	移動到資料夾(OpenVINO-YOLOV4-master)下
Cd C:\Users\...\Desktop\OpenVINO-YOLOV4-master

4.	將 .weights 轉 .pb (以tiny為例)
先將.weights放入/OpenVINO-YOLOV4-master，再將我github的.names放入/OpenVINO-YOLOV4-master/cfg，再執行:
```
python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny.weights --data_format NHWC –tiny
```

執行成功後就會在資料夾看到多一個.pb檔

5.	透過官方套件將.pb轉IR
在使官方套件執行轉IR時，必須先設定環境參數，設定的方法如下:
```
"C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"
```
執行以上指令就會設定你現在所使用的CMD的參數，讓有OpenVINO的程式跑得起來，如果把CMD關掉，下次再開新的CMD視窗要再重新執行這段命令。
另外要將/OpenVINO-YOLOV4-master中，yolo_v4_tiny.json的classes改成自己的訓練數目(我訓練4類，所以classes:4)，其他不用更改。
如果是要轉yolo_v4就要用yolo_v4.json，classes也一樣要改。
設定完就可以用以下指令把.pb轉IR格式了~
```
python "C:\Program Files (x86)\Intel\openvino_2021.4.689\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolo_v4_tiny.json --batch 1 --reverse_input_channels
```
***3. 利用 OpenVINO 進行 demo***
前提:您必須在本地端(桌機/筆電...)安裝OpenVINO
我使用的OpenVINO版本:2021.4
安裝過程可參考:
https://medium.com/ching-i/intel-openvino%E4%BB%8B%E7%B4%B9%E8%88%87%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-15b07473d998
![image](https://user-images.githubusercontent.com/101262636/173179243-7c1ee78e-0b63-4f45-99a3-735864b94432.png)

我使用C:\ProgramFiles(x86)\Intel\openvino_2021.4.689\deployment_tools\inference_engine\demos\object_detection_demo\python 的 object_detection_demo.py 做 demo

(1)轉移到OpenVINO的object_detection_demo目錄底下
```
cd C:\Program Files (x86)\Intel\openvino_2021.4.689\deployment_tools\inference_engine\demos\object_detection_demo\python
```
(2) 使用object_detection_demo.py執行推論
```
python object_detection_demo.py -m C:\Users\mcvlab\Desktop\OpenVINO-YOLOV4-master\frozen_darknet_yolov4_model.xml 
-at yolo 
-i 你要demo的影片位置(如: C:/.../.../XX.mp4)
-d GPU (不輸入這行就是用CPU執行，還可選CPU/MYRAIAD(NCS2，要先裝驅動，win10會自動抓))
--labels label的位置(如: C:/.../.../XX.names)(將我放在github的資料下載下來)
```
### 測試結果
#### 於colab測試
![](https://i.imgur.com/An3bQLZ.jpg)

![](https://i.imgur.com/7VKoVqQ.jpg)

![](https://i.imgur.com/NyIiksI.jpg)

![](https://i.imgur.com/aZ2NEyf.jpg)
#### 利用OpenVINO測試於本地端
https://drive.google.com/file/d/1OV2wYQwCHXCMKcxMIMOxiR2vOGI2On5-/view?usp=sharing

### 免責聲明
此專案用於學術研討，請勿用於不當行為及商業行為，如有侵權請通知本人，謝謝。
