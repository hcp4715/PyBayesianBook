# PyBayesianBook

## 关于本项目
本项目将用于将《贝叶斯统计及其在Python中实现》一课中的录音转为jupyterbook，以便于后续的整理和学习。

课程相关信息：[Gitee仓库](https://gitee.com/hcp4715/bayesian-analysis-nnupsy) | [B站视频](https://www.bilibili.com/video/BV1W6paeLExS/)


## 关于作者

讲者：胡传鹏(博士)，南京师范大学心理学院教授、博士生导师, [实验室网站](huchuanpeng.com).

本课程录音的转录及电子书的制作由如下同学协助
- 朱珊珊，南京师范大学心理学院在读硕士生
- 潘晚坷，南京师范大学心理学院在读博士生
- 王继贤，哈尔滨师范大学心理学专业在读硕士生
- 陈星宇，浙江师范大学心理学专业在读硕士生
- 高东宇，宁波大学心理学专业在读硕士生
- 张易泓，华南师范大学特殊教育专业在读硕士生
- 陈晨，浙江师范大学心理学专业在读本科生
- 朱田牧 浙江理工大学工程心理学专业硕士毕业生
## 许可
本项目中的所有内容，除课程相关的资料外，均基于[CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)协议共享。

## 如何贡献

本项目采用[github](https://github.com/hcp4715/PyBayesianBook)进行管理，欢迎大家提交PR。具体而言，在`book`这一文件中，有各个章节的md文件以及图片。如果发现哪个章节有问题，找到与之对应的md文件进行修改后在github上提交PR即可。步骤如下：

0. 注册github账号（如果还没有的话）
1. fork本项目
2. 在本地克隆你自己的仓库
3. 在本地仓库中新建一个分支
4. 修改有问题的章节所对应的md文件
5. commit后push到你自己的仓库
6. 在github上提交一个PR

## 本项目文件目录

```
.
├── README.md
├── book
|   ├── _config.yml
|   ├── _toc.yml
│   ├── _build
│   |   └── html
│   ├── chapter_overview
│   │   ├── intro.md
│   │   ├── Part1.md
│   │   ├── Part2.md
│   │   └── Part3.md
|   ├── chpater1_Bayesrules
|   |   ├── Bayesrule_part1.md
|   |   ├── Bayesrule_part2.md
|   |   └── Bayesrule_part3.md

```

<!--- 如何使用GitHub pages来发布电子书：
https://jupyterbook.org/en/stable/start/publish.html

1. 使用`jb build book`命令生成html文件
jupyter-book build book/   

or

jb build book/ 

2. 使用`ghp-import`命令将html文件发布到github上
cd ./book
ghp-import -n -p -f _build/html
->