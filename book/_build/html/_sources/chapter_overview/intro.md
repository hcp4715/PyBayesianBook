# 序言
<center>

<font size=5>**高级心理统计 (Advanced Statistics in Psych Sci)**</font>

**《贝叶斯统计及其在Python中的实现》 (Bayesian inference with Python)**

**Instructor： 胡传鹏（博士）(Dr. Hu Chuan-Peng)**

**南京师范大学心理学院 (School of Psychology, Nanjing Normal University)**
</center>



<font size=3>&emsp;&emsp;首先，感谢屏幕前的各位能够来到这里学习这门课程。虽然这门课叫做高级心理统计，但内容不一定与其他的《高级心理统计学》相同。总的来看，这门课程主要是以贝叶斯统计为核心，后续也会介绍一些比较复杂的线性模型以及实现这些模型分析的代码。因此，这可能和大家预想的高级心理统计不太一样。</font>


## 1、学员构成

<font size=3>&emsp;&emsp;这是我第三次在南京师范大学上这个高级心理统计课程，或者叫做贝叶斯推断、贝叶斯统计。今年我终于有勇气地进行对外开放。因此大家可以看到我们课题组的公众号上有一个报名链接，以及让校外的参与者填写问卷。</font>

<font size=3>&emsp;&emsp;以下是校外参与学员的构成情况：</font>
<font size=3>

- **学生身份**
<center><img width = '800' height ='120' src="image-2.png"></center>

- **教育程度**
<center><img width = '800' height ='200' src="image-3.png"></center>

- **专业领域**
<center><img width = '400' height ='300' src="image-4.png"></center>

- **编程经验**
<center><img width = '600' height ='300' src="image-5.png"></center>
</font>
<br>

- <font size=3> 总结：绝大部分都是拥有心理学、教育学等背景的高学历在读学生，但普遍缺乏编程基础。
- 建议：屏幕前的各位完全不用担心缺乏编程经验，在这个信息开放时代，只需要在网上学习几个小时的在线课程就可以拥有后续进行数据分析的编程基础。</font>
<br>

## 2、课程大纲

<font size=4>**Intro**</font>

1: 课程介绍（为什么要用贝叶斯/PyMC3，展示一个贝叶斯因子和贝叶斯层级模型例子，课程安排）

<font size=4>**I Basics：**</font>

2: 贝叶斯公式
- 单一事件的贝叶斯模型(先验、似然、分母和后验)
- 随机变量的贝叶斯模型(先验、似然、分母和后验)
- 
3: 建立一个完整的贝叶斯模型：Beta-Binomial model

- Beta先验
- Binomial与似然
- Beta-Binomial model

4: 贝叶斯模型的特点：数据-先验与动态更新

- 先验与数据对后验的影响
- 数据顺序不影响后验

5: 经典的贝叶斯模型：Conjugate families

- Gamma-Poisson
- Normal-Normal
  
<font size=4>**II 近似后验估计**</font>

6: 近似后验的方法

- 网格法
- MCMC

7: 深入一种MCMC算法

- M-H 算法

8: 基于后验的推断

- 后验估计
- 假设检验
- 后验预设

<font size=4>**III Bayesian回归模型**</font>

9: 简单的Bayesian线性回归模型

- 建立模型
- 调整先验
- 近似后验
- 基于近似后验的推断
- 序列分析

10: 对回归模型的评估

- 评估模型的标准
- 对简单线性模型的估计

11: 扩展的线性模型

- 多自变量的线性回归

12: GLM: Bayes factors

13: GLM: Logistic Regression

14: Hierachical Bayesian Model 1

15: Hierachical Bayesian Model 2
