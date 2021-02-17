# AI_edge_4
第４回AIエッジコンテスト (team:ymym)　成果物

# ファイルの説明

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
|  
|-Makefile  
