"""
Basic operations on trees.
"""

import numpy as np
from collections import defaultdict

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.parent_dep_type = None
        self.num_children = 0
        self.children = list()
        self.idx = None
        self.dis = None

    def add_child(self, child, dep_type=None):
        
        self.num_children += 1
        if dep_type is None:
            child.parent = self
            self.children.append(child)
        else:
            child.parent = self
            child.parent_dep_type = dep_type
            self.children.append((child, dep_type))

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_, prune, subj_pos, obj_pos, maxlen):
    """
    Convert a sequence of head indexes into a tree object.
    """
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None


    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = -1 # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h-1].add_child(nodes[i])

    # find dependency path
    subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
    obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

    cas = None

    subj_ancestors = set(subj_pos)
    for s in subj_pos:
        h = head[s]
        tmp = [s]
        while h > 0:
            tmp += [h - 1]
            subj_ancestors.add(h - 1)
            h = head[h - 1]

        if cas is None:
            cas = set(tmp)
        else:
            cas.intersection_update(tmp)

    obj_ancestors = set(obj_pos)
    for o in obj_pos:
        h = head[o]
        tmp = [o]
        while h > 0:
            tmp += [h - 1]
            obj_ancestors.add(h - 1)
            h = head[h - 1]
        cas.intersection_update(tmp)

    # find lowest common ancestor
    if len(cas) == 1:
        lca = list(cas)[0]
    else:
        child_count = {k: 0 for k in cas}
        for ca in cas:
            if head[ca] > 0 and head[ca] - 1 in cas:
                child_count[head[ca] - 1] += 1

        # the LCA has no child in the CA set
        for ca in cas:
            if child_count[ca] == 0:
                lca = ca
                break

    path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
    path_nodes.add(lca)

    # compute distance to path_nodes
    dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

    for i in range(len_):
        if dist[i] < 0:
            stack = [i]
            while stack[-1] >= 0 and stack[-1] not in path_nodes:
                stack.append(head[stack[-1]] - 1)

            if stack[-1] in path_nodes:
                for d, j in enumerate(reversed(stack)):
                    dist[j] = d
            else:
                for j in stack:
                    if j >= 0 and dist[j] < 0:
                        dist[j] = int(1e4)  # aka infinity

    # 不可及点为编码0， 其余的 + 1
    for i in range(len(dist)):
        d = dist[i]
        assert d != -1
        if d != 1e4:
            dist[i] = d+1
        else:
            dist[i] = 0
    while len(dist) < maxlen:
        dist.append(0)
    assert len(dist) == maxlen
    assert root is not None
    return root, dist


def tree_to_adj(sent_len, tree):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)
    # tree是一个可迭代的对象 深度优先遍历
    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]
        children=[]
        for c in t.children:
            ret[t.idx, c.idx] = 1
            children.append(c)
            
        queue += children

    return ret


def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret