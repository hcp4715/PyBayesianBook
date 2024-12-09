# 序言
<center>

<font size=5>**高级心理统计 (Advanced Statistics in Psych Sci)**</font>

**《贝叶斯统计及其在Python中的实现》 (Bayesian inference with Python)**

**Instructor： 胡传鹏（博士）(Dr. Hu Chuan-Peng)**

**南京师范大学心理学院 (School of Psychology, Nanjing Normal University)**
</center>



<font size=3>&emsp;&emsp;首先，感谢大家阅读本电子材料。这门课在学校的系统中叫做《高级心理统计》，但内容以贝叶斯统计为核心，是在贝叶斯统计的框架下介绍如何实现常见的统计分析，包括一些比较复杂的线性模型。</font>


## 1、课程大纲

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
