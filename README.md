# 自分で実装した決定木

やる

## ログ

https://www.slideshare.net/takemikami/r13-9821987
http://codecrafthouse.jp/p/2014/09/decision-tree/
http://darden.hatenablog.com/entry/2016/12/15/222447

これらを参考にやる


何から作ろうか.


* gini係数とか計算式の実装からやる？ -> やろう -> できた
* 次再帰処理やる ->  できた
* グラフ化やる -> できた
* predictする -> できた
* max_depthを実装 -> できた
* entropyの実装 -> できた
* カテゴリカルな値と連続値を両方扱えるようにする -> できてた
* 分割数を 2 以上にできるようにする

## predict_probaの実装まとめ

ざっくり調べて実装したところ.

* 訓練時のサンプル数を元に、確率値を出す.

### 訓練時のサンプル数

1. 入力されたデータ1つが到達するnodeを探す
2. nodeの節であれば次へ(left of right)
3. nodeの葉であれば, 学習時に訓練データのクラスごとの個数があるはずなのでその情報を残しておく

### ↑で返ってきたデータを確率値に修正する

1. labelが0,1だった場合、データの形式は[[7,6],[3,4]]こうなるはず.
2. これをaxis=1を軸にしてsumする
3. それで割る -> すると, うまく分類された方のclassが1.0に近い確率になる.
