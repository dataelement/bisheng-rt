"""
###############################################################################
# Copyright Info :    Copyright (c) Dataelem Institute. All rights reserved.
# Filename       :    post_cell.py
# Abstract       :    Post processing of table. Get the format html output.

# Current Version:    0.0.1
# Date           :    2022-06-16
###############################################################################
"""

from html import escape

import numpy as np
from networkx import Graph, find_cliques


def area_to_html(area, labels, texts_tokens):
    """ Generate structure html and text tokens from area, which is the
        intermediate result of post-processing.

    Args:
        area(np.array): (n x n). Two-dimensional array representing the
                        distribution of rows and columns for each cell.
        labels(list[list]): (n x 1). labels of each non-empty cell
        texts_tokens(list[list]): texts_tokens for each non-empty cell

    Returns:
        list(str): The html that characterizes the structure of table
        list(str): text tokens for each cell (including empty cells)
    """

    area_extend = np.zeros([area.shape[0] + 1, area.shape[1] + 1])
    area_extend[:-1, :-1] = area
    html_struct_recon = []
    text_tokens_recon = []
    headend = 0
    for height in range(area.shape[0]):
        html_struct_recon.append('<tr>')
        width = 0
        numhead, numbody = 0, 0
        while width < area.shape[1]:
            # curent cell is rest part of a rowspan cell
            if height != 0 and area_extend[height,
                                           width] == area_extend[height - 1,
                                                                 width]:
                width += 1

            # td without span
            elif (area_extend[height, width] != area_extend[height + 1, width]
                  and area_extend[height, width] != area_extend[height,
                                                                width + 1]):
                html_struct_recon.append('<td>')
                html_struct_recon.append('</td>')
                texts_insert = texts_tokens[
                    int(area_extend[height, width]) -
                    1] if int(area_extend[height, width]) >= 1 else ['']
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    # empty cell
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += 1

            # td only with colspan
            elif area_extend[height, width] != area_extend[
                    height + 1, width] and area_extend[
                        height, width] == area_extend[height, width + 1]:
                colspan = 1
                while area_extend[height,
                                  width] == area_extend[height,
                                                        width + colspan]:
                    colspan += 1
                    if (width + colspan) == (area.shape[1]):
                        break
                html_struct_recon.append('<td')
                html_struct_recon.append(" colspan=\"%s\"" % str(colspan))
                html_struct_recon.append('>')
                html_struct_recon.append('</td>')
                texts_insert = texts_tokens[
                    int(area_extend[height, width]) -
                    1] if int(area_extend[height, width]) >= 1 else ['']
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += colspan

            # td only with rowspan
            elif area_extend[height, width] == area_extend[
                    height + 1, width] and area_extend[
                        height, width] != area_extend[height, width + 1]:
                rowspan = 1
                while area_extend[height,
                                  width] == area_extend[height + rowspan,
                                                        width]:
                    rowspan += 1
                    if height + rowspan == area.shape[0]:
                        break
                html_struct_recon.append('<td')
                html_struct_recon.append(" rowspan=\"%s\"" % str(rowspan))
                html_struct_recon.append('>')
                html_struct_recon.append('</td>')
                texts_insert = texts_tokens[
                    int(area_extend[height, width]) -
                    1] if int(area_extend[height, width]) >= 1 else ['']
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += 1

            # td with row span and col span togther
            elif area_extend[height, width] == area_extend[
                    height + 1,
                    width] and area_extend[height,
                                           width] == area_extend[height,
                                                                 width + 1]:
                rowspan = 1
                while area_extend[height,
                                  width] == area_extend[height + rowspan,
                                                        width]:
                    rowspan += 1
                    if height + rowspan == area.shape[0]:
                        break
                html_struct_recon.append('<td')
                html_struct_recon.append(" rowspan=\"%s\"" % str(rowspan))
                colspan = 1
                while area_extend[height,
                                  width] == area_extend[height,
                                                        width + colspan]:
                    colspan += 1
                    if (width + colspan) == (area.shape[1]):
                        break
                html_struct_recon.append(" colspan=\"%s\"" % str(colspan))
                html_struct_recon.append('>')
                html_struct_recon.append('</td>')
                texts_insert = texts_tokens[
                    int(area_extend[height, width]) -
                    1] if int(area_extend[height, width]) >= 1 else ['']
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += colspan

        html_struct_recon.append('</tr>')
        if numhead > numbody:
            headend = height + 1

    # insert '<thead>', '</thead>', '<tbody>' and '</tbody>'
    rowindex = [
        ind for ind, td in enumerate(html_struct_recon) if td == '</tr>'
    ]
    if headend:
        html_struct_recon.insert(rowindex[headend - 1] + 1, '</thead>')
        html_struct_recon.insert(rowindex[headend - 1] + 2, '<tbody>')
    else:
        # 默认第一行为标题
        trindex = html_struct_recon.index('</tr>')
        html_struct_recon.insert(trindex + 1, '</thead>')
        html_struct_recon.insert(trindex + 2, '<tbody>')
    html_struct_recon.insert(0, '<thead>')
    html_struct_recon.append('</tbody>')

    return html_struct_recon, text_tokens_recon


def format_html(html_struct, text_tokens):
    """ Formats HTML code from structure html and text tokens

    Args:
        html_struct (list(str)): structure html
        text_tokens (list(dict)): text tokens

    Returns:
        str: The final html of table.
    """

    html_code = html_struct.copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], text_tokens[::-1]):
        if cell['tokens']:
            cell = [
                escape(token) if len(token) == 1 else token
                for token in cell['tokens']
            ]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)
    html_code = ''.join(html_code)
    # html_code = '''<html><body><table>%s</table></body></html>''' % html_code
    html_code = '''<table>%s</table>''' % html_code

    return html_code


def bbox2adj(bboxes_non):
    """Calculating row and column adjacent relationships according to bboxes of
       non-empty aligned cells

    Args:
        bboxes_non(np.array): (n x 4).bboxes of non-empty aligned cells

    Returns:
        np.array: (n x n).row adjacent relationships of non-empty aligned cells
        np.array: (n x n).column adjacent relationships of non-empty
                  aligned cells
    """

    adjr = np.zeros([bboxes_non.shape[0], bboxes_non.shape[0]], dtype='int')
    adjc = np.zeros([bboxes_non.shape[0], bboxes_non.shape[0]], dtype='int')
    x_middle = bboxes_non[:, ::2].mean(axis=1)
    y_middle = bboxes_non[:, 1::2].mean(axis=1)
    min_cell_x = min(bboxes_non[:, 2] - bboxes_non[:, 0])
    min_cell_y = min(bboxes_non[:, 3] - bboxes_non[:, 1])
    width_cell = abs(bboxes_non[:, 2] - bboxes_non[:, 0])
    for i, box in enumerate(bboxes_non):
        indexr = np.where((bboxes_non[:, 1] < y_middle[i])
                          & (bboxes_non[:, 3] > y_middle[i]))[0]
        indexc = np.where((bboxes_non[:, 0] < x_middle[i])
                          & (bboxes_non[:, 2] > x_middle[i]))[0]
        adjr[indexr, i], adjr[i, indexr] = 1, 1
        adjc[indexc, i], adjc[i, indexc] = 1, 1

        # Determine if there are special row relationship
        for j, box2 in enumerate(bboxes_non):
            if not (box2[1] + min_cell_y / 4 >= box[3]
                    or box[1] + min_cell_y / 4 >= box2[3]):
                # box1和box2在y上有交集（且重叠区域大于最小单元格/4），交集内部包含其他
                # box在y轴上的中点，则box1和box2有联系
                # if not (box2[1] + 4 >= box[3] or box[1] + 4 >= box2[3]):
                #     # box1和box2在y上有交集（且交集大于4个像素），且交集内部包含其他box在
                #     # y轴上的中点，则box1和box2有联系
                indexr2 = np.where((max(box[1], box2[1]) < y_middle[:])
                                   & (y_middle[:] < min(box[3], box2[3])))[0]
                if len(indexr2):
                    adjr[j, i], adjr[i, j] = 1, 1

        # Determine if there are special column relationship
        for j, box2 in enumerate(bboxes_non):
            if not (box2[0] + min_cell_x / 4 >= box[2]
                    or box[0] + min_cell_x / 4 >= box2[2]):
                # box1和box2在x上有交集（且重叠区域大于最小单元格/4），交集内部包含其他
                # box在x轴上的中点，则box1和box2有联系
                # if not (box2[0] + 8 >= box[2] or box[0] + 8 >= box2[2]):
                #     # box1和box2在x上有交集（且交集大于4个像素），且交集内部包含其他box在
                #     # x轴上的中点，则box1和box2有联系
                indexc2 = np.where((max(box[0], box2[0]) < x_middle[:])
                                   & (x_middle[:] < min(box[2], box2[2])))[0]
                if len(indexc2):
                    thr = min(abs(box[2] - box2[0]), abs(box2[2] - box[0]))
                    cur_min_cell = min(width_cell[indexc2])
                    # 重叠区域大于最小cell宽度/2.5，则相邻
                    if cur_min_cell < 2.5 * thr:
                        adjc[j, i], adjc[i, j] = 1, 1

    return adjr, adjc


def rect_max_iou(box_1, box_2):
    """Calculate the maximum IoU between two boxes: the intersect area / the
       area of the smaller box

    Args:
        box_1 (np.array | list): [x1, y1, x2, y2]
        box_2 (np.array | list): [x1, y1, x2, y2]

    Returns:
        float: maximum IoU between the two boxes
    """

    addone = 0  # 0 in mmdet2.0 / 1 in mmdet 1.0
    box_1, box_2 = np.array(box_1), np.array(box_2)

    x_start = np.maximum(box_1[0], box_2[0])
    y_start = np.maximum(box_1[1], box_2[1])
    x_end = np.minimum(box_1[2], box_2[2])
    y_end = np.minimum(box_1[3], box_2[3])

    area1 = (box_1[2] - box_1[0] + addone) * (box_1[3] - box_1[1] + addone)
    area2 = (box_2[2] - box_2[0] + addone) * (box_2[3] - box_2[1] + addone)
    overlap = np.maximum(x_end - x_start + addone, 0) * np.maximum(
        y_end - y_start + addone, 0)

    return overlap / min(area1, area2)


def nms_inter_classes(bboxes, iou_thres=0.3):
    """NMS between all classes

    Args:
        bboxes(list): [bboxes in cls1(np.array), bboxes in cls2(np.array), ].
                       bboxes of each classes
        iou_thres(float): nsm threshold

    Returns:
        np.array: (n x 4).bboxes of targets after NMS between all classes
        list(list): (n x 1).labels of targets after NMS between all classes
    """

    lable_id = 0
    merge_bboxes, merge_labels = [], []
    for bboxes_cls in bboxes:
        if lable_id:
            merge_bboxes = np.concatenate((merge_bboxes, bboxes_cls), axis=0)
        else:
            merge_bboxes = bboxes_cls
        merge_labels += [[lable_id]] * len(bboxes_cls)
        lable_id += 1

    mark = np.ones(len(merge_bboxes), dtype=int)
    score_index = merge_bboxes[:, -1].argsort()[::-1]
    for i, cur in enumerate(score_index):
        if mark[cur] == 0:
            continue
        for ind in score_index[i + 1:]:
            if mark[ind] == 1 and rect_max_iou(merge_bboxes[cur],
                                               merge_bboxes[ind]) >= iou_thres:
                mark[ind] = 0
    new_bboxes = merge_bboxes[mark == 1, :4]
    new_labels = np.array(merge_labels)[mark == 1]
    new_labels = [list(map(int, lab)) for lab in new_labels]

    return new_bboxes, new_labels


def adj_to_cell(adj, bboxes, mod):
    """Calculating start and end row / column of each cell according to
       row / column adjacent relationships

    Args:
        adj(np.array): (n x n). adjacent relationships of non-empty cells
        bboxes(np.array): (n x 4). bboxes of non-empty aligned cells
        mod(str): 'row' or 'col'

    Returns:
        list(np.array): start and end row of each cell if mod is 'row' / start
                        and end col of each cell if mod is 'col'
    """

    assert mod in ('row', 'col')

    # generate graph of each non-empty aligned cells
    nodenum = adj.shape[0]
    edge_temp = np.where(adj != 0)
    edge = list(zip(edge_temp[0], edge_temp[1]))
    table_graph = Graph()
    table_graph.add_nodes_from(list(range(nodenum)))
    table_graph.add_edges_from(edge)

    # Find maximal clique in the graph
    clique_list = list(find_cliques(table_graph))

    # Sorting the maximal cliques
    coord = []
    times = np.zeros(nodenum)
    for clique in clique_list:
        for node in clique:
            times[node] += 1
    for ind, clique in enumerate(clique_list):
        # The nodes that only belong to this maximal clique will be selected
        # to order, unless all nodes in this maximal clique belong to multi
        # maximal clique
        nodes_nospan = [node for node in clique if times[node] == 1]
        # nodes_select = nodes_nospan if len(nodes_nospan) else clique
        if len(nodes_nospan):
            nodes_select = nodes_nospan
        else:
            # todo: 目前横跨所有行/列的合并单元格不参与计算，后续可以进一步优化
            nodes_select = [
                node for node in clique if times[node] < len(clique_list)
            ]
        coord_mean = [
            ind, (bboxes[nodes_select, 1] + bboxes[nodes_select, 3]).mean()
        ] if mod == 'row' else [
            ind, (bboxes[nodes_select, 0] + bboxes[nodes_select, 2]).mean()
        ]
        coord.append(coord_mean)
    coord = np.array(coord, dtype='int')
    # Sorting the maximal cliques according to coordinate mean of nodes_select
    coord = coord[coord[:, 1].argsort()]

    # Start and end row of each cell if mod is 'row' / start and end col of
    # each cell if mod is 'col'
    listcell = [[] for _ in range(nodenum)]
    for ind, coo in enumerate(coord[:, 0]):
        for node in clique_list[coo]:
            listcell[node] = np.append(listcell[node], ind)

    return listcell


def ocr_result_matching(cell_bboxes, ocr_results, iou_thres=0.7, sep_char=''):
    """ assign ocr_results to cells acrroding to their position

    Args:
        cell_bboxes(list): (n x 4). aligned bboxes of non-empty cells
        ocr_results(dict): ocr results of the table including bboxes of
                           single-line text and therir content
        iou_thres(float): matching threshold between bboxes of cells and bboxes
                          of single-line text

    Returns:
        list(str): ocr results of each cell
    """
    texts_assigned = []
    cells_matched_bboxes, cells_matched_texts, cells_matched_orders = \
        [], [], []
    ocr_bboxes, ocr_texts, ocr_orders = ocr_results['bboxes'], ocr_results[
        'texts'], ocr_results['orders']

    delta_y_threhold = 5
    for i, box_cell in enumerate(cell_bboxes):
        matched_bboxes, matched_texts, matched_orders = [], [], []
        for j, box_text in enumerate(ocr_bboxes):
            if rect_max_iou(box_cell, box_text) >= iou_thres:
                # Insert curent ocr bbox into the matched_bboxes list according
                # to Y coordinate,
                # Changelog: add same row judge with delta_y threshold
                if len(matched_bboxes) == 0:
                    matched_bboxes.append(box_text)
                    matched_texts.append(ocr_texts[j])
                    matched_orders.append(ocr_orders[j])
                else:
                    insert_staus = 0
                    for k, matched_box in enumerate(matched_bboxes):
                        delta_y = np.abs(box_text[1] - matched_box[1])
                        if delta_y <= delta_y_threhold:
                            continue

                        if box_text[1] < matched_box[1]:
                            matched_bboxes.insert(k, box_text)
                            matched_texts.insert(k, ocr_texts[j])
                            matched_orders.insert(k, ocr_orders[j])
                            insert_staus = 1
                            break
                    if not insert_staus:
                        matched_bboxes.append(box_text)
                        matched_texts.append(ocr_texts[j])
                        matched_orders.append(ocr_orders[j])

        # Get the ocr result of the current cell
        # matched_texts = [txt for txt in matched_texts if len(txt)]
        if len(matched_texts) == 0:
            texts_assigned.append('')
        elif len(matched_texts) == 1:
            texts_assigned.append(matched_texts[0])
        else:
            merge = matched_texts[0]
            for txt in matched_texts[1:]:
                if txt and merge and (txt[0] != '%' and merge[-1] != '-'):
                    # merge += " "
                    merge += sep_char
                merge += txt
            texts_assigned.append(merge)

        cells_matched_bboxes.append(matched_bboxes)
        cells_matched_texts.append(matched_texts)
        cells_matched_orders.append(matched_orders)

    return texts_assigned, cells_matched_bboxes, \
        cells_matched_texts, cells_matched_orders


class PostCell():
    """Get the format html of table
    """
    def __init__(
        self,
        nms_inter=False,
        nms_threshold=0.3,
    ):
        """
        Args:
            nms_inter(bool): whether using nms inter classes.
            nms_threshold(float): nsm threshold
        """

        super().__init__()
        self.nms_inter = nms_inter
        self.nms_threshold = nms_threshold

    def post_processing(self, batch_result, ocr_results=None, sep_char=''):
        """
        Args:
            batch_result(list(Tensor)): prediction results,
                like [box_result, ...]

        Returns:
            list(str): Format results, [html of table1, html of table2, ...]
        """
        table_results = []
        for rid, result in enumerate(batch_result):
            table_result = {'html': '', 'content_ann': {}}
            # Processing bboxes of aligned cells, such as nms between all
            # classes and bboxes refined according to lgpma
            bboxes_results = result

            if self.nms_inter:
                bboxes, labels = nms_inter_classes(bboxes_results,
                                                   self.nms_threshold)
                labels = [[lab[0]] for lab in labels]
            else:
                bboxes, labels = bboxes_results[0], [[0]] * len(
                    bboxes_results[0])
                for cls in range(1, len(bboxes_results)):
                    bboxes = np.concatenate((bboxes, bboxes_results[cls]),
                                            axis=0)
                    labels += [[cls]] * len(bboxes_results[cls])

            # Return empty result, if processed bboxes of aligned cells empty.
            if not len(labels):
                table_results.append({
                    'html': '',
                    'bboxes': [],
                    'labels': [],
                    'texts': [],
                    'cells': [],
                    'cells_matched_bboxes': [],
                    'cells_matched_texts': [],
                    'cells_matched_orders': []
                })
                continue

            # If ocr results are provided, assign them to corresponding cell
            bboxes = [list(map(round, b[0:4])) for b in bboxes]
            if ocr_results is None:
                texts = [''] * len(bboxes)
                cells_matched_bboxes = [[]] * len(bboxes)
                cells_matched_texts = [[]] * len(bboxes)
                cells_matched_orders = [[]] * len(bboxes)
            else:
                match_res = ocr_result_matching(bboxes,
                                                ocr_results[rid],
                                                sep_char=sep_char)
                texts = match_res[0]
                cells_matched_bboxes = match_res[1]
                cells_matched_texts = match_res[2]
                cells_matched_orders = match_res[3]

            # Calculating cell adjacency matrix according to bboxes of
            # non-empty aligned cells
            bboxes_np = np.array(bboxes)
            adjr, adjc = bbox2adj(bboxes_np)

            # Predicting start and end row / column of each cell according to
            # the cell adjacency matrix
            colspan = adj_to_cell(adjc, bboxes_np, 'col')
            rowspan = adj_to_cell(adjr, bboxes_np, 'row')
            cells = [[row.min(), col.min(),
                      row.max(), col.max()]
                     for col, row in zip(colspan, rowspan)]
            cells = [list(map(int, cell)) for cell in cells]
            cells_np = np.array(cells)

            # Searching empty cells and recording them through arearec
            arearec = np.zeros(
                [cells_np[:, 2].max() + 1, cells_np[:, 3].max() + 1])
            for cellid, rec in enumerate(cells_np):
                srow, scol, erow, ecol = rec[0], rec[1], rec[2], rec[3]
                arearec[srow:erow + 1, scol:ecol + 1] = cellid + 1
            empty_index = -1  # deal with empty cell
            for row in range(arearec.shape[0]):
                for col in range(arearec.shape[1]):
                    if arearec[row, col] == 0:
                        cells.append([row, col, row, col])
                        arearec[row, col] = empty_index
                        empty_index -= 1

            # Generate html of each table.
            html_str_rec, html_text_rec = area_to_html(arearec, labels, texts)
            table_result['html'] = format_html(html_str_rec, html_text_rec)

            # Append empty cells and sort all cells so that cells information
            # are in the same order with html
            num_empty = len(cells) - len(bboxes)
            if num_empty:
                bboxes += [[]] * num_empty
                labels += [[]] * num_empty
                texts += [''] * num_empty
                cells_matched_bboxes += [[]] * num_empty
                cells_matched_texts += [[]] * num_empty
                cells_matched_orders += [[]] * num_empty
            sortindex = np.lexsort(
                [np.array(cells)[:, 1],
                 np.array(cells)[:, 0]])
            bboxes = [bboxes[i] for i in sortindex]
            labels = [labels[i] for i in sortindex]
            texts = [texts[i] for i in sortindex]
            cells = [cells[i] for i in sortindex]
            cells_matched_bboxes = [cells_matched_bboxes[i] for i in sortindex]
            cells_matched_texts = [cells_matched_texts[i] for i in sortindex]
            cells_matched_orders = [cells_matched_orders[i] for i in sortindex]
            # only two class: 0 head, 1 body
            if len(bboxes_results) == 2:
                head_s, head_e = html_str_rec.index(
                    '<thead>'), html_str_rec.index('<tbody>')
                head_num = html_str_rec[head_s:head_e + 1].count('</td>')
                labels = [[0]] * head_num + [[1]] * (len(cells) - head_num)

            table_result['content_ann'] = {
                'bboxes': bboxes,
                'labels': labels,
                'texts': texts,
                'cells': cells,
                'cells_matched_bboxes': cells_matched_bboxes,
                'cells_matched_texts': cells_matched_texts,
                'cells_matched_orders': cells_matched_orders
            }
            table_results.append(table_result)

        return table_results
