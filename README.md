# jittor-reimplement-wiki2bio

This is the project opened by tqx19 and mingk19 in Tsinghua University. This is another implementation of wiki2bio project which is also on github.

这是清华大学学生唐启勋和明可建立的仓库，是对于GitHub上已有项目wiki2bio（<https://github.com/tyliupku/wiki2bio>）的另一种jittor实现。

The dataset can be found on the original project.

数据集可在原项目下下载。

We try to change the code from tensorflow1.0.0 and python2.7 to jittor1.3.1 and python3.7.

我们将代码从python2.7下的tf框架1.0.0版本改为python3.7下的jittor框架1.3.1

As the original code do not declare a lisence, we choose to use GNU lisence to ensure the code to be open-sourced.

因原有仓库代码未做开源声明，我们采用GNU协议来保证代码的开源。

The download link of Jittor is <https://cg.cs.tsinghua.edu.cn/jittor/>

Jittor框架的下载链接如上

Compared to the original code we change a bug occured in preprocess.py and modified its LSTM according to standard implementation. Also ,we added a fgateGRU layer to test whether it's better than LSTM.  We delete original beam-search to simplify our code.

相较于原有代码，我们修复了一处preprocess.py中出现的bug，将LSTM调整为标准实现，并且，我们提供了一个Field-gating 的 GRU层来进行对比。我们删除了beam-search部分来简化代码。

As the web demo provided by mingk19 is close-sourced, we won't provide its code here. You may ask to mingk19 for it.  

因为明可提供的WEB demo是闭源的，我们此处并不提供其代码。你可以尝试找明可要。

We provide a pre-trained model which name is model.pkl , you can just use it to test this model.

我们提供了一个名为model.pkl的预训练模型，可用于进行测试。

You should just download the dataset from the link provided above, and unzip it , then you should just place the folder into the folder you place the code. You should first run preprocess.py and then Main.py.

你可以在上文提供的链接里面下载数据集，将其解压，放到你放代码的文件夹中。你应该首先运行preprocess.py，然后运行Main.py.
