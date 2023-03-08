# LeNet-5

LeNet-5 第一版架構未進行任何改善及優化</br>
activation funtion 有改為 relu</br>
此版中的C3層由tensorflow中的高階API構成，並未使用論文中原本的連接方式
</br>
</br>
預測時間大約落在0.036s ~ 0.040s之間</br>
準確率極低，即使寫得很清楚也很常判斷錯誤
</br>

## 加入Image Augmentation

由於先前雖然evaluate得準確度極高，但在實際手寫做辨識上的正確率卻不高於20%，因此試著加入image augmentation來增加訓練資料。</br>
</br>
雖然在加入後實際手寫辨識的準確度大幅提高，但在training accuracy與validation accuracy 卻出現了奇怪的現象。validation accuracy為 9X%而training accuracy卻只有7X%
</br>

<img src='./images/accuracy_40_0.3_0.5_0.5.png' style="width:400px">
