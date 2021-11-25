# #N叉树层序遍历
# class Solution:
#     # #递归法
#     # def levelOrder(self, root):
#     #     if not root:
#     #         return []
#     #
#     #     res = [root.val]
#     #     def dfs(index, root):
#     #         if len(res) < index:
#     #             res.append([])
#     #         res[index - 1].append(root.val)
#     #         if root.children:
#     #             n = len(root.children)
#     #             for i in range(n):
#     #                 dfs(index+1, root.children[i])
#     #
#     #
#     #     dfs(1, root)
#     #     return res
#
#     #迭代法
#     def levelOrder(self, root):
#         if not root:
#             return []
#
#         res = []
#         queue = [root]
#         while queue:
#             numNodes = len(queue)
#             tmp = []
#             for _ in range(numNodes):
#                 cur = queue.pop()
#                 tmp.append(cur.val)
#                 if cur.children:
#                     for i in range(cur.children):
#                         queue.append(cur.children[i])
#             res.append(tmp)
#
#         return res


#二叉树的层序遍历
class Solution:
    #递归法
    def levelOrder(self, root):
        if not root:
            return []
        res = []
        def dfs(index, node):
            if len(res) < index:
                res.append([])
            res[index-1].append(node.val)
            if root.left:
                dfs(index+1, node.left)
            if root.right:
                dfs(index+1, node.right)
        dfs(1, root)
        return res
