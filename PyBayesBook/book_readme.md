# 本地加载电子书的方式

- 首先，需要下载整个PyBayesBook文件。
- 进入到Bayes_book/_build/html,点击index.html即可进入电子书。

# 在线加载电子书的方式

- 总的来说，是通过python的“ghp-import”包将电子书的HTML文件夹上传至github，同时自动创建一个网址来展示电子书。

- 具体步骤
（1）将ghp-import包安装到相应环境
（2）将github库clone到本地并完成电子书的修改和添加，然后将本地完整的电子书文件push到github的库里
（3）打开Anaconda powershell prompt或者其他可以运行python包的命令窗口，进入到相应环境。
（4）cd到第2步里电子书的父目录，注意：此处应当包含.git配置文件
（5）运行下面的代码
```python
ghp-import -n -p -f 电子书文件名称/_build/html
```
（6）ghp-import包会自动帮助你创建一个“gh-pages”的分支（注意：不需要提前创建，这里是自动创建的），该分支能够储存html文件并建立一个网址来放置它们。
（7）进入到github里，点击刚才创建的分支“gh-pages”，页面的右侧会出现一个“Deployments”，进入后会发现创建了一个新的网址，这就是在线电子书的网址。