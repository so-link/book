# debug技巧

> 转载请注明作者[梦里茶](https://github.com/ahangchen)


- 看crash 堆栈，有堆栈可以直接知道哪里出了问题
- 查日志，日志里的蛛丝马迹很重要，没有的话就要补日志，补日志要注意按多个可能的条件分支去补
- 重现，不管是看堆栈还是查日志，一定要重现出来，并且在修改后不能按之前的路径重现才算解决了bug
- 版本分析：看bug是在哪个版本出现的，为什么之前的版本没有，相邻两次之间修改了什么