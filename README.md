# AI_edge_4
第４回AIエッジコンテスト (team:ymym)　成果物

# ファイルの説明
下記は、TensorFlow(TF)で学習したモデルpbから実行ファイルを生成するまでに使ったファイル  
xilinxのコードを参考に実装  
https://github.com/Xilinx/Vitis-AI/tree/v1.1/Tool-Example  
  
↓Vitis AIにより量子化、コンパイルしたファイル一式  
deeplab  
　　|  
　　|-build  
　　|　　|-dputils.o  
　　|　　|-main.o  
　　|  
　　|-float  
　　|　　|-frozen_model.pb　　　　　　：浮動小数点の凍結された推論グラフ  
　　|  
　　|-src  
　　|　　|-main.cpp　　　　　　　　　：アプリのソースコード  
　　|  
　　|-vai_c_output_custom  
　　|　　|-deeplab_kernel.info　　　　：VAI_C カーネル　情報ファイル  
　　|　　|-deeplab_kernel_graph.gv　　：VAI_C カーネル　トポロジ記述ファイル  
　　|　　|-dpu_deeplab.elf　　　　　　：VAI_C カーネル　本体  
　　|  
　　|-vai_q_output  
　　|　　|-deploy_model.pb　　　　　：VAI コンパイラ用の量子化されたモデル  
　　|　　|-quantize_eval_model.pb　　：評価用の量子化されたモデル  
　　|  
　　|-deeplab　　　　　　　　　　　：Makefileで生成した実行ファイル  
　　|-Makefile  
  
  
↓量子化、コンパイルに使ったツール一式  
（下記フォルダ内に上記deeplabフォルダを置いて作業、xilinx提供のツールを利用）  
workspace  
　　|-4_tf_quantize.sh　　　　　　　　　：量子化スクリプト(TF用)  
　　|-6_tf_compile_for_v2.sh　　　　　　：コンパイルスクリプト(TF用)  
　　|-custom.json  
　　|-deeplab_eval.py　　　　　　　　　：quantize_eval_model.pb評価用  
　　|-dpu-03-26-2020-13-30.dcf  
　　|-ultra96v2_oob.hwh　　　　　　　：ハードウェア情報ファイル  
    
    
