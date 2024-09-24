# Part 1: Why Bayesian Inference ?</font>

<font size=6>思考：</font>

- <font size=5>研究人类心理和行为的规律，容易吗？

- 相比之下，发论文容易吗？</font>

## 1、如何定义人类的心理和行为规律

<font size=4>**1.1&emsp;心理学的定义</font>**

<font size=3>&emsp;&emsp;我们心理学一般从个体角度来去研究人类的心理与行为，因此绝大部分研究都是从一个单独的个体入手，即便他是在一定的社会文化历史条件之下，那么他的行为是什么样的？

&emsp;&emsp;再比如说，在发展心理学当中，重点是研究人类的心理与行为如何发展？社会心理学当中，个体的心理与行为是如何受到社会情境的影响？组织心理学当中，主要是研究如何提高组织的效率？由此看来，不同领域的定义不同，当我们在定义的时候会发现心理与行为规律的概念本身就很复杂，而选择其中一个定义去研究，又是一个很复杂的问题。</font>

<font size=4>**1.2&emsp;其他领域</font>**
<font size=3>

&emsp;&emsp;下面这张图片是Science创刊125周年的封面，它列出了125个重要但还未被完全回答的问题。其中，最重要的两个问题分别是：1、宇宙的起源 &emsp; 2、意识的生物学基础
</font>
<center><img width = '230' height ='310' src="image-6.png"></center>

[Source: https://www.science.org/toc/science/309/5731](https://www.science.org/toc/science/309/5731)

<font size=4>
Q1: What is the Uiverse Made of ? (physics)

**<font color=#FF8C00>Q2: What is the Biological Basis of Consciouness ? (psychological/cognitive science)
</font>**

## 2、如何研究——研究的方法和工具
思考：重要性或复杂性相似的问题，是否意味着研究方法也趋于复杂 ？

**2.1 &emsp;物理学中的方法 (Methods in Physics)**

Example 1: Webb telescope (韦伯望远镜) (equipment)

Example 2: Big-team science (CERN, the European Organization for Nuclear Research) ---- equipment & practices

Example 3: Mathematics
<center><img width = '370' height ='210' src="image-7.png"></center>

<br>

**2.2 &emsp;其他研究人类智能的领域所采用的方法 (Methods in other fields that also study "intelligence")**

Example: Artificial Intelligence (e.g., Chat-gpt)
<center><img width = '370' height ='210' src="image-8.png"></center>
<br>

**2.3 &emsp;心理科学的研究方法 (What do psychological scientists have?)**

*你们能够想到的研究方法有哪些？*
<center><img width = '400' height ='210' src="image-9.png"></center>

**实证研究：**
- 质性研究
- 观察法
- 问卷
- 行为实验
- 眼动、生理数据记录
- EEG/ERP/MEG
- fMRI/PET/fNIRs
- TMS/tDCS
- ...

**统计方法：**
- t-test
- ANOVA
- Correlation
-Structural equation model (SEM)
- ?

**相关方法课程：**
- 心理测量
- 心理统计（包括SPSS等）
- 实验心理学（包括E-prime等）
- ？

**针对更复杂的数据**
- 数据字化的时代，大数据
- 神经成像/生理数据
- 多模态的数据融合
- ...

**如果要研究更复杂的问题，我们可能需要：**
- 更好的仪器
- <font color=#FF8C00>**更好的统计/数据分析**</font>
- 更好的实践 (e.g., big-team science)

## 3、一种更好的统计方法

<font size=5>**贝叶斯统计 (Bayesian statistics)**</font>

<br>
<center><img width = '600' height ='320' src="image-10.png"></center>

[Souce: https://www.nature.com/articles/s43586-020-00001-2](https://www.nature.com/articles/s43586-020-00001-2)

<font size=5>**贝叶斯统计的优势**</font>

**1、通用/灵活/强大**
- 通用：贝叶斯推断在多个学科特别是AI领域得到了广泛应用。
- 灵活：结合先验知识与新数据，灵活调整模型以适应不同数据情境。
- 强大：与频率主义学派相比，贝叶斯方法适用范围更广。

**2、相对易用**

- 概率编程语言(Probabilistic Programming Languages, PPLs)的发展和普及使得研究者可以相对快速地掌握贝叶斯统计
  
&emsp;&emsp;PPLs: computational languages for statistical modeling

&emsp;&emsp;e.g.
- Python (PyMC、NumPyro)
- R (Stan、JAGS)
- BUGS
- Julia (Mamba、Turing.jl)
- ...

**3、可拓展**
- 贝叶斯概念已经应用到以深度学习为中心的新技术的发展，包括深度学习框架(TensorFlow, Pytorch)，创建能力更强、数据驱动的模型。

**4、方便交流**
- 大部分PPLs都有类似的数据结构，但是不同的学科使用的语言不同。

- 心理学/社会科学/神经科学：

&emsp;&emsp;PyMC (bambi)

&emsp;&emsp;Stan (brms)

&emsp;&emsp;BUGS

*大部分情况下，开发者使用它可以轻松地定义概率模型，然后程序会自动地求解模型*
<center><img width = '600' height ='420' src="image-11.png"></center>

[Source: https://towardsdatascience.com/intro-to-probabilistic-programming-b47c4e926ec5](https://towardsdatascience.com/intro-to-probabilistic-programming-b47c4e926ec5)

