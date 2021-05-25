import re, sys

NOUN_SET = set([
    'NNP',
    'NNPS',
    'PRP$',
    'NP',
    'NML',
])


class TreeNode(object):
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []


def build_tree(tree_tokens):
    # base case: return terminal node
    if len(tree_tokens) == 1:
        return TreeNode(tree_tokens[0])

    # recursive case: recursively build all children
    assert tree_tokens[0] == '('
    assert tree_tokens[-1] == ')'
    parent = TreeNode(tree_tokens[1])

    stack = []
    curr_child_tokens = []
    for token in tree_tokens[2:-1]:

        if token == '(':
            stack.append(token)
        elif token == ')':
            stack.pop()
        curr_child_tokens.append(token)

        if not stack:
            child = build_tree(curr_child_tokens)
            parent.children.append(child)
            child.parent = parent
            curr_child_tokens = []

    return parent


def tokenize_tree_string(tree_string):
    tokens = []
    curr_string = []
    for char in tree_string:
        if char == '(' or char == ')':
            if curr_string:
                tokens.append(''.join(curr_string))
                curr_string = []
            tokens.append(char)
        elif not char.strip() and curr_string:
            tokens.append(''.join(curr_string))
            curr_string = []
        elif char.strip():
            curr_string.append(char)

    # sanity checks
    assert tokens[0] == '('
    assert tokens[-1] == ')'

    return tokens


def build_trees_from_path(path):
    curr_lines = []
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    stack = []
    trees = []

    for line in lines:

        line = line.strip()
        if not line:
            continue

        for char in line:
            if char == '(':
                stack.append(char)
            elif char == ')':
                stack.pop()
        curr_lines.append(line)

        # if we reached the end of a parse tree...
        if not stack:
            tree_string = ' '.join(curr_lines)
            tree_tokens = tokenize_tree_string(tree_string)

            root = build_tree(tree_tokens)
            trees.append(root)
            curr_lines = []

    return trees


def annotate_tree(root, ctr=0):
    if not root:
        return

    # terminal node
    if not root.children:
        root.idx = ctr
        ctr += 1
        return ctr

    for child in root.children:
        ctr = annotate_tree(child, ctr)

    return ctr


def get_leaf_descendents(root, array=None):
    if array is None:
        array = []

    # terminal
    if not root.children:
        array.append(root)

    for child in root.children:
        get_leaf_descendents(child, array)

    return array


def get_noun_nonterminals(root, array=None, noun_set=None):
    if noun_set is None:
        noun_set = NOUN_SET

    if array is None:
        array = []

    # terminal
    if not root.children:
        return array

    # otherwise, nonterminal.  if noun, add to list.
    if root.value in noun_set:
        array.append(root)

    for child in root.children:
        get_noun_nonterminals(child, array, noun_set=noun_set)

    return array


if __name__ == '__main__':
    INCLUDE_NER = True

    assert len(sys.argv) == 3

    trees = build_trees_from_path(sys.argv[1])
    output_file = sys.argv[2]

    all_spans = []
    for tree in trees:
        annotate_tree(tree)
        all_tokens = [i.value for i in get_leaf_descendents(tree)]
        nps = get_noun_nonterminals(tree)
        spans = set()
        for np in nps:
            nodes = get_leaf_descendents(np)
            span = (nodes[0].idx, nodes[-1].idx)
            spans.add(span)
        all_spans.append((all_tokens, spans))

    def detokenize(tokens):
        punct=['-','.',',',':',';',"'s"] # will need to remove leading space
        subst={'-LRB-':'(', '-RRB-':')', '``':'"', "''":'"', '$ ':'$'}
        index_map=[None for _ in range(sum(len(x) for x in tokens))]
        tok_c,detok_c=0,0
        detokenized_string=""
        for jj,tok in enumerate(tokens):
            if tok in subst:
                new=subst[tok]
                if (new==')') or len(detokenized_string)==0 or (tok=="''"):
                    for ii in range(len(tok)):
                        index_map[tok_c+ii]=detok_c
                    detokenized_string+=new
                    detok_c+=len(new)
                else:
                    for ii in range(len(tok)):
                        index_map[tok_c+ii]=detok_c+1
                    detokenized_string+=" "+new
                    detok_c+=len(new)+1
                tok_c+=len(tok)
            elif (tok in punct) or (len(detokenized_string)==0) \
                    or (detokenized_string[-1]=='(') or (tokens[jj-1]=='``'):
                detokenized_string+=tok
                for ii in range(len(tok)):
                    index_map[tok_c+ii]=detok_c+ii
                detok_c+=len(tok)
                tok_c+=len(tok)
            else:
                detokenized_string+=" "+tok
                for ii in range(len(tok)):
                    index_map[tok_c+ii]=detok_c+ii+1
                detok_c+=len(tok)+1
                tok_c+=len(tok)
        assert all(x is not None for x in index_map)
        return detokenized_string, index_map

    def to_char(word_spans):
        tokens,spans=word_spans
        char_spans=[None for _ in spans]
        for ii,(w_start,w_end) in enumerate(spans):
            c_start=sum(len(x) for x in tokens[:w_start])
            # indexing is inclusive as far as i can tell from the rest of the code
            c_end=sum(len(x) for x in tokens[:w_end+1])-1 # -1 to be inclusive
            char_spans[ii]=(c_start,c_end)
        return char_spans

    def to_str(x):
        # x[0] is the sentence, x[1] are the spans
        output_str,char_map=detokenize(x[0])
        char_spans=to_char(x)
        # very basic sanity checks
        assert len(char_spans)==len(x[1])
        # this means that if span1 starts after span2 in word space, it does so in character space as well
        for word_span1,char_span1 in zip(x[1],char_spans):
            for word_span2,char_span2 in zip(x[1],char_spans):
                det_start=(word_span1[0]-word_span2[0])*(char_span1[0]-char_span2[0])
                assert det_start>=0
                det_end=(word_span1[1]-word_span2[1])*(char_span1[1]-char_span2[1])
                assert det_end>=0
        str_spans=[]
        for ii,y in enumerate(char_spans):
            start,end=y
            mapped_start=char_map[start]
            mapped_end=char_map[end]
            str_spans.append(str(mapped_start)+'-'+str(mapped_end)+'-'+str(ii))
        output_str+='\n'+' '.join(str_spans)
        return str(output_str)

    with open(output_file, 'w') as f:
        f.write('\n'.join(map(to_str, all_spans))+'\n')
