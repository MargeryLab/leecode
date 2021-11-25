import math
# #编辑距离
# class Solution:
#     # #解法一，二维数组
#     # def minDistance(self, word1: str, word2: str) -> int:
#     #     #0要指代”空串”而不是第一个字符，所以要多一位
#     #     n1 = len(word1)
#     #     n2 = len(word2)
#     #     dp = [[0] * (n2+1) for _ in range(n1+1)]
#     #     #边界
#     #     for i in range(n1+1):
#     #         dp[i][0] = i
#     #     for j in range(n2+1):
#     #         dp[0][j] = j
#     #     for i in range(1, n1+1):
#     #         for j in range(1, n2+1):
#     #             dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + int(word1[i-1] != word2[j-1]))
#     #
#     #     return dp[n1][n2]
#
#     #解法二，一维数组
#     def minDistance(self, word1: str, word2: str) -> int:
#         m = len(word1)
#         n = len(word2)
#         dp = list(range(n+1))
#         for i in range(m):
#             lu = dp[0]
#             dp[0] = i+1
#             for j in range(n):
#                 dp[j + 1], lu = min(dp[j] + 1, dp[j + 1] + 1, lu + int(word1[i] != word2[j])), dp[j + 1]
#         return dp[-1]


#完全平方数
# class node:
#     def __init__(self, value, step=0):
#         self.value = value
#         self.step = step
#
#     # def __str__(self):
#     #     return '<value:{}, step:{}>'.format(self.value, self.step)
#
# class Solution:
    # #解法一：递归
    # def numSquares(self, n: int) -> int:
    #     import math
    #     res = n-int(math.sqrt(n))**2+1
    #     def square(n, cnt):
    #         nonlocal res
    #         if cnt >= res:
    #             return res
    #         if n // math.sqrt(n) == math.sqrt(n):
    #             res = cnt + 1
    #             return res
    #
    #         base = int(math.sqrt(n))
    #         for i in range(base, base//2, -1):
    #             square(n-i**2, cnt+1)
    #         return res
    #
    #     return square(n, 0)

    # """
    # 解法二：广度优先遍历 BFS
    # BFS 3元素：
    # 队列(先进先出), 入队出队的节点，已访问的集合
    # """
    # def numSquares(self, n: int) -> int:
    #     #1、初始化三元素
    #     Node = node(n)#根节点
    #     queue = [Node]
    #     visited = set([Node.value])
    #
    #     while True:
    #         #2、操作队列:弹出队首节点
    #         vertex = queue.pop(0)
    #         #3、操作弹出的节点：根据业务生成子节点（一个或多个）
    #         residuals = [vertex.value - n*n for n in range(1, int(vertex.value**.5)+1)]
    #         for i in residuals:
    #             new_vertex = node(i, vertex.step+1)
    #             if i == 0:
    #                 return new_vertex.step
    #             elif i not in visited:
    #                 queue.append(new_vertex)
    #                 visited.add(i)
    #
    #     return -1

    # #动态规划
    # def numSquares(self, n: int) -> int:
    #     # """
    #     # d[i] 表示最少需要多少个数的平方来表示整数i。
    #     # 状态转移方程：
    #     # """
    #     # nums = [i * i for i in range(1, int(n ** 0.5) + 1)]
    #     # f = [0] + [float('inf')] * n
    #     # for num in nums:
    #     #     for j in range(num, n + 1):
    #     #         f[j] = min(f[j], f[j - num] + 1)
    #     # return f[-1]
    #
    #     #贪心算法
    #     ps = set([i * i for i in range(1, int(n ** 0.5) + 1)])
    #     def divisible(n, count):
    #         if count == 1: return n in ps
    #         for p in ps:
    #             if divisible(n - p, count - 1):
    #                 return True
    #         return False
    #
    #     for count in range(1, n + 1):
    #         if divisible(n, count):
    #             return count

# class Solution:
    # #解法一：使用标记数组 时间复杂度：O(mn), 空间复杂度：O(m+n)
    # def setZeroes(self, matrix):
    #     """
    #     Do not return anything, modify matrix in-place instead.
    #     """
    #     m, n = len(matrix), len(matrix[0])
    #     row, col = [False] * m, [False] * n
    #
    #     for i in range(m):
    #         for j in range(n):
    #             if matrix[i][j] == 0:
    #                 row[i] = col[j] = True
    #
    #     for i in range(m):
    #         for j in range(n):
    #             if row[i] or col[j]:
    #                 matrix[i][j] = 0

    # #使用两个标记向量 时间复杂度：O(mn), 空间复杂度：O(1)
    # def setZeroes(self, matrix):
    #     """
    #     Do not return anything, modify matrix in-place instead.
    #     """
    #     m, n = len(matrix), len(matrix[0])  # 由第一行和第一列存储每行每列是否有0
    #     row1, col1 = False, False
    #     for i in range(m):
    #         for j in range(n):
    #             if i == 0 or j == 0:
    #                 if matrix[0][j] == 0:
    #                     row1 = True
    #                 if matrix[i][0] == 0:
    #                     col1 = True
    #             elif matrix[i][j] == 0:
    #                 matrix[0][j] = matrix[i][0] = 0
    #
    #     for i in range(1, m):
    #         for j in range(1, n):
    #             if matrix[i][0] == 0 or matrix[0][j] == 0:
    #                 matrix[i][j] = 0
    #
    #     if row1:
    #         for j in range(n):
    #             matrix[0][j] = 0
    #
    #     if col1:
    #         for i in range(m):
    #             matrix[i][0] = 0
    #
    #     return matrix

    # #解法三：使用一个标记向量, 只使用一个标记变量记录第一列是否原本存在0,这样，第一列的第一个元素即可以标记第一行是否出现0
    # def setZeroes(self, matrix):
    #     col1 = False
    #     m, n = len(matrix), len(matrix[0])
    #     for i in range(m):
    #         if matrix[i][0] == 0:
    #             col1 = True
    #         for j in range(1, n):
    #             if matrix[i][j] == 0:
    #                 matrix[i][0] = matrix[0][j] = 0
    #
    #     for i in range(m-1, -1, -1):
    #         for j in range(1, n):
    #             if matrix[i][0] == 0 or matrix[0][j] == 0:
    #                 matrix[i][j] = 0
    #         if col1:
    #             matrix[i][0] = 0
    #
    #     return matrix

# #200 岛屿数量
# class Solution:
#     # 方法一：深度优先搜索 DFS
#     def dfs(self, grid, i, j):
#         grid[i][j] = '0'
#         nr, nc = len(grid), len(grid[0])
#         for r, c in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
#             if 0 <= r < nr and 0 <= c < nc and grid[r][c] == '1':
#                 self.dfs(grid, r, c)
#
#     def numIslands(self, grid):
#         row = len(grid)
#         col = len(grid[0])
#         count = 0
#         for i in range(row):
#             for j in range(col):
#                 if grid[i][j] == '1':
#                     self.dfs(grid,i, j)
#                     count += 1
#
#         return count

    # # 方法二：广度优先搜索 BFS
    # def numIslands(self, grid):
    #     nr = len(grid)
    #     nc = len(grid[0])
    #     def bfs(i, j):
    #         queue = [[i, j]]
    #         while queue:
    #             [i ,j] = queue.pop(0)
    #             if 0 <= i < nr and 0 <= j < nc and grid[i][j] == '1':
    #                 grid[i][j] = '0'#表示访问过了
    #                 queue += [[i-1, j], [i+1, j], [i, j-1], [i, j+1]]#上下左右
    #
    #     count = 0
    #     for i in range(nr):
    #         for j in range(nc):
    #             if grid[i][j] == '0': continue
    #             bfs(i, j)
    #             count += 1
    #
    #     return count

# #102. 二叉树的层序遍历
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
# class Solution:
#     # #迭代法
#     # def levelOrder(self, root):
#     #     res = []
#     #     if not root:
#     #         return []
#     #     queue = [root]
#     #     while queue:
#     #         #获取队列长度，相当于当前这一层的节点个数
#     #         size = len(queue)
#     #         tmp = []
#     #         for _ in range(size):
#     #             r = queue.pop(0)
#     #             tmp.append(r.val)
#     #             if r.left:
#     #                 queue.append(r.left)
#     #             if r.right:
#     #                 queue.append(r.right)
#     #         res.append(tmp)
#     #
#     #     return res
#
#     # 递归实现
#     def levelOrder(self, root):
#         if not root:
#             return []
#         res = []
#         def dfs(index, r): #当前层及当前遍历根节点
#             if len(res) < index:
#                 res.append([])
#             res[index-1].append(r.val)
#             if r.left:
#                 dfs(index+1, r.left)
#             if r.right:
#                 dfs(index+1, r.right)
#
#         dfs(1, root)#为什么不是从0开始
#         return res


# #剑指 Offer II 001. 整数除法
# class Solution:#循环使用a-b，计算减了多少次，即可得到结果
#     # 超时
#     """
#     2.
#     a = 22, b = 3              k=1 value=3
#         22 - (3+3)=16          k=2 value=6
#         22 - (6+6)=10          k=4 value=12
#         22 - (12+12)<0
#     a = 10, b=3                k=1
#         10-(3+3)=4             k=2
#         10-(6+6)<0
#     a = 4, b=3                 k=1
#         4-(3+3)<0
#     a = 1, b=3
#     3.
#     22-(3<<31) < 0
#     22-(3<<30) < 0
#     ...
#     22-(3<<3) = -2 < 0
#     22-(3<<2) = 10 > 0   k=1<<2=4
#     10-(3<<1) = 4 > 0    k=1<<1=2
#     4-(3<<0) = 1 > 0     k=1<<0=1
#
#     """
#     def divide(self, a: int, b: int) -> int:
#         INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1
#         if a == INT_MIN and b == -1:
#             return INT_MAX
#         if a == INT_MAX and b ==1:
#             return INT_MAX
#
#         #都>0或都<0，sign=1, sign = -1说明有一个为负数     同为0，异为1
#         sign = -1 if (a > 0) ^ (b > 0) else 1 #如果a、b两个值不相同，则异或结果为1。如果a、b两个值相同，异或结果为0。
#         if a > 0:
#             a = -a
#         if b > 0:
#             b = -b
#
#         ans = 0
#         while a <= b: #正的时候是a>b
#             value = b #bei jian shu
#             k = 1
#             #0xc0000000 means -2^30, is -2^31/2
#             while (a - (value+value)) >= 0 and value>= 0xc0000000:
#                 k += k
#                 value += value
#             ans, a = ans+k, a-value
#
#         # bug 修复：因为不能使用乘号，所以将乘号换成三目运算符
#         return ans*sign


# #二叉树的hou序遍历
# class Solution:
#     def postorderTraversal(self, root):  # 左-右-打印
#         # #递归
#         # res = []
#         # def dfs(root):
#         #     if not root:
#         #         return
#         #     dfs(root.left)
#         #     dfs(root.right)
#         #     res.append(root.val)
#
#         # dfs(root)
#         # return res
#
#         # 迭代
#         res = []
#         stack = []
#         while stack or root:
#             # 不断往左子树方向走，每走一次就将当前节点保存到栈中
#             # 这是模拟递归的调用
#             if root:
#                 stack.append(root)
#                 root = root.left
#             # 当前节点为空，说明左边走到头了，从栈中弹出节点并保存
#             # 然后转向右边节点，继续上面整个过程
#             else:
#                 tmp = stack.pop()
#                 res.append(tmp.val)
#                 root = tmp.right


# #二叉树的preorder遍历
# class Solution():
#     def preorderTraversal(self, root):#root-left-right
#         res, stack = [], [root]
#         while root or stack:
#             while root:
#                 res.append(root.val)
#                 stack.append(root)
#                 root = root.left
#             root = stack.pop()
#             root = root.right
#         return res


# #二叉树的inorder遍历
# class Solution():
#     def preorderTraversal(self, root):#left-root-right
#         res, stack = [], []
#         while root or stack:
#             while root:
#                 stack.append(root)
#                 root = root.left
#             root = stack.pop()
#             res.append(root.val)
#             root = root.right
#         return res


# #二叉树的postorder遍历
# class Solution():
#     def preorderTraversal(self, root):#left-right-root,依次将根节点、右孩子、左孩子入栈
#         res, stack = [], [] #stack同时存放了访问过左子树和访问过右子树的结点
#         prev = None #用于区分栈顶结点的右子树是否已经遍历过
#         while root or stack:
#             while root:
#                 stack.append(root)
#                 root = root.left
#             # 比前序和中序的模板增加一个判断过程：节点没有右孩子或已经访问了该节点的子节点
#             if not root.right or root.right == prev:
#                 res.append(root.val)
#                 prev = root
#                 root = None
#             else:
#                 stack.append(root)
#                 root = root.right
#         return res


# #N cha tree postorder
# class Solution:
#     def postorder(self, root):
#         # # 递归
#         # res = []
#         #
#         # def dfs(root):
#         #     for child in root.children:
#         #         dfs(child)
#         #     res.append(root.val)
#         #
#         # if root:
#         #     dfs(root)
#         # return res
#
#         # iteration
#         res, stack =[], []
#         cur = root
#         while stack or cur:
#             while cur:
#                 stack.append((cur, 0))
#                 if cur.children:
#                     cur = cur.children[0]
#                 else:
#                     cur = None
#             cur, c = stack.pop() #c 记录该结点有多少个孩子结点已经遍历过
#             if not cur.children or c >= len(cur.children):
#                 res.append(cur.value)
#                 cur = None
#             else:
#                 c += 1
#                 stack.append((cur, c))
#                 if c < len(cur.children):
#                     cur = cur.children[c]
#                 else:
#                     cur = None
#         return res

# #剑指 Offer II 046. 二叉树的右侧视图
# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
# class Solution:
#     def rightSideView(self, root: TreeNode):
#         res = []
#         def dfs(index, root):
#             if root == None:
#                 return
#             if index == len(res):
#                 res.append(root.val)
#             dfs(index+1, root.right)
#             dfs(index+1, root.left)
#
#         dfs(0, root)
#
#         return res

# class Solution:
#     def findDiagonalOrder(self, nums):
#         rec = [[]]
#         for i in range(len(nums)):
#             for j in range(len(nums[i])):
#                 # i+j = len(Rec)-1 = rec的index
#                 if (i+j) >= len(rec):
#                     rec.append([])
#                 rec[i + j].append(nums[i][j])
#
#         res = []
#         for i in range(len(rec)):
#             res += rec[i][::-1]
#         return res
#

#回文串
# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         x = str(x)
#         size = len(x)
#         if size % 2 == 0:
#             if x[size//2-1] != x[size//2]:
#                 return False
#             str1 = x[:(size//2)-1]
#             str2 = x[(size//2+1):]
#         else:
#             mid = size // 2
#             str1 = x[:mid]
#             str2 = x[(mid+1):]
#
#         if str1 == str2[::-1]:
#             return True
#         else:
#             return False


#LCP 12. 小张刷题计划
class Solution:
    def minTime(self, time, m) -> int:
        l,r = 0,sum(time)
        while l<r:
            mid = (l+r)>>1  #右移一位即除2
            if self.check(mid,time,m):
                r = mid
            else:
                l = mid + 1
        return l

    def check(self, limit, cost, day):
        use_day,total_time,max_time = 1,0,cost[0]
        for i in cost[1:]:
            if total_time+min(max_time,i)<= limit:
                total_time,max_time = total_time+min(max_time,i),max(max_time,i)
            else:
                use_day += 1
                total_time,max_time = 0,i
        return use_day<=day


if __name__ == '__main__':
    s = Solution()
    # res = s.minDistance("horse", "ros")
    # res = s.numSquares(24)
    # res = s.setZeroes([[0,1,2,0],[3,4,5,2],[1,3,1,5]])
    # res = s.numIslands([["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]])
    # res = s.divide(7, -3)
    # res = s.levelOrder([3,9,20,None,None,15,7])
    # res = s.rightSideView([1,2,3,4])
    # res = s.findDiagonalOrder([[1,2,3],[4],[5,6,7],[8],[9,10,11]])
    # res = s.isPalindrome(-101)
    res = s.minTime(time = [1,2,3,3], m = 2)
    print(res)