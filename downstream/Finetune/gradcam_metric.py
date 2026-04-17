import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import pandas as pd
from typing import List, Tuple, Dict, Any
import ipdb
from sklearn.metrics import auc

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的IoU
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        IoU值
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def match_boxes_per_image(pred_boxes: List, gt_boxes: List, iou_threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    单张图片内的框匹配
    Args:
        pred_boxes: 预测框列表 [[x1, y1, x2, y2], ...]
        gt_boxes: 真实框列表 [[x1, y1, x2, y2], ...]
        iou_threshold: IoU阈值
    Returns:
        (TP, FP, FN)
    """
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0
    
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)  # 只有FN
    
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0  # 只有FP
    
    # 构建IoU矩阵
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(np.array(pred), np.array(gt))
    
    # 贪心匹配
    pred_matched = [False] * len(pred_boxes)
    gt_matched = [False] * len(gt_boxes)
    
    # 按IoU从高到低排序所有可能的匹配
    all_matches = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                all_matches.append((iou_matrix[i, j], i, j))
    
    all_matches.sort(reverse=True)
    
    tp = 0
    for iou, i, j in all_matches:
        if not pred_matched[i] and not gt_matched[j]:
            tp += 1
            pred_matched[i] = True
            gt_matched[j] = True
    
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    
    return tp, fp, fn

def evaluate_detection_dataset_per_class(all_pred_boxes: Dict, all_gt_boxes: Dict, 
                                       class_names: List[str], iou_thresholds: List[float] = None) -> Dict[str, Any]:
    """
    评估整个数据集的目标检测性能（按类别）
    Args:
        all_pred_boxes: 每张图片每个类别的预测框字典 {class_name: {image_id: [[x1, y1, x2, y2], ...]}, ...}
        all_gt_boxes: 每张图片每个类别的真实框字典 {class_name: {image_id: [[x1, y1, x2, y2], ...]}, ...}
        class_names: 类别名称列表
        iou_thresholds: IoU阈值列表
    Returns:
        评估结果字典，包含每个类别和平均指标
    """
    if iou_thresholds is None:
        iou_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    results = {}
    
    # 评估每个类别
    for class_name in class_names:
        class_pred_boxes = all_pred_boxes.get(class_name, {})
        class_gt_boxes = all_gt_boxes.get(class_name, {})
        
        class_results = {}
        
        for iou_threshold in iou_thresholds:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_pred = 0
            total_gt = 0
            
            # 获取该类别的所有图片ID
            all_image_ids = set(list(class_pred_boxes.keys()) + list(class_gt_boxes.keys()))
            
            # 逐图片计算
            for image_id in all_image_ids:
                pred_boxes = class_pred_boxes.get(image_id, [])
                gt_boxes = class_gt_boxes.get(image_id, [])
                
                tp, fp, fn = match_boxes_per_image(pred_boxes, gt_boxes, iou_threshold)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_pred += len(pred_boxes)
                total_gt += len(gt_boxes)
            
            # 计算指标
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
            
            class_results[iou_threshold] = {
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'num_pred': total_pred,
                'num_gt': total_gt
            }
        
        results[class_name] = class_results
    
    # 计算平均指标
    results['average'] = {}
    for iou_threshold in iou_thresholds:
        avg_precision = np.mean([results[cls][iou_threshold]['precision'] for cls in class_names])
        avg_recall = np.mean([results[cls][iou_threshold]['recall'] for cls in class_names])
        avg_f1 = np.mean([results[cls][iou_threshold]['f1_score'] for cls in class_names])
        avg_accuracy = np.mean([results[cls][iou_threshold]['accuracy'] for cls in class_names])
        
        results['average'][iou_threshold] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'accuracy': avg_accuracy,
            'num_pred': sum([results[cls][iou_threshold]['num_pred'] for cls in class_names]),
            'num_gt': sum([results[cls][iou_threshold]['num_gt'] for cls in class_names]),
            'tp': sum([results[cls][iou_threshold]['tp'] for cls in class_names]),
            'fp': sum([results[cls][iou_threshold]['fp'] for cls in class_names]),
            'fn': sum([results[cls][iou_threshold]['fn'] for cls in class_names])
        }
    
    return results

def calculate_map_per_class(results: Dict, class_names: List[str]) -> Dict[str, float]:
    """
    计算每个类别的mAP
    Args:
        results: 评估结果字典
        class_names: 类别名称列表
    Returns:
        每个类别的mAP字典
    """
    map_results = {}
    
    for class_name in class_names + ['average']:
        if class_name in results:
            aps = [result['precision'] for result in results[class_name].values()]
            map_results[class_name] = np.mean(aps) if aps else 0.0
    
    return map_results

def calculate_ap(precisions: List[float], recalls: List[float]) -> float:
    """
    计算AP (Average Precision) - PR曲线下面积
    Args:
        precisions: 精确率列表
        recalls: 召回率列表
    Returns:
        AP值
    """
    # 确保召回率从0到1单调递增
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))
    
    # 对精确率进行单调递减处理
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # 计算AP（PR曲线下面积）
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap

def evaluate_detection_with_confidence(all_pred_boxes_with_conf: Dict, all_gt_boxes: Dict, 
                                      class_names: List[str], iou_thresholds: List[float] = None) -> Dict[str, Any]:
    """
    使用置信度计算AP和mAP（标准目标检测评估方法）
    Args:
        all_pred_boxes_with_conf: 包含置信度的预测框 {class: {image_id: [[x1,y1,x2,y2,conf], ...]}}
        all_gt_boxes: 真实框 {class: {image_id: [[x1,y1,x2,y2], ...]}}
        class_names: 类别名称列表
        iou_thresholds: IoU阈值列表
    Returns:
        包含AP和mAP的评估结果
    """
    if iou_thresholds is None:
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    ap_results = {}
    
    for iou_threshold in iou_thresholds:
        ap_results[iou_threshold] = {}
        
        for class_name in class_names + ['average']:
            if class_name == 'average':
                # 跳过平均值的单独计算，后面会计算
                continue
                
            # 收集该类别的所有预测框（带置信度）和真实框
            all_predictions = []
            all_gts = []
            
            # 收集预测框
            if class_name in all_pred_boxes_with_conf:
                for image_id, pred_boxes in all_pred_boxes_with_conf[class_name].items():
                    for pred_box in pred_boxes:
                        if len(pred_box) >= 5:  # 确保有置信度
                            x1, y1, x2, y2, conf = pred_box[:5]
                            all_predictions.append({
                                'image_id': image_id,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf
                            })
            
            # 收集真实框
            if class_name in all_gt_boxes:
                for image_id, gt_boxes in all_gt_boxes[class_name].items():
                    for gt_box in gt_boxes:
                        all_gts.append({
                            'image_id': image_id,
                            'bbox': gt_box
                        })
            
            # 如果没有预测或真实框，AP为0
            if not all_predictions or not all_gts:
                ap_results[iou_threshold][class_name] = {
                    'ap': 0.0,
                    'precisions': [],
                    'recalls': [],
                    'num_predictions': len(all_predictions),
                    'num_gts': len(all_gts)
                }
                continue
            
            # 按置信度降序排序预测框
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # 初始化TP和FP数组
            tp = np.zeros(len(all_predictions))
            fp = np.zeros(len(all_predictions))
            
            # 为每个真实框创建匹配记录（按图片分组）
            gt_matches = defaultdict(list)
            for gt in all_gts:
                gt_matches[gt['image_id']].append({'gt': gt, 'matched': False})
            
            # 遍历每个预测框（按置信度从高到低）
            for i, pred in enumerate(all_predictions):
                image_id = pred['image_id']
                pred_bbox = pred['bbox']
                
                best_iou = 0
                best_gt_idx = -1
                
                # 在同一图片中寻找匹配的真实框
                if image_id in gt_matches:
                    for j, gt_match in enumerate(gt_matches[image_id]):
                        if gt_match['matched']:
                            continue
                        
                        gt_bbox = gt_match['gt']['bbox']
                        iou = calculate_iou(np.array(pred_bbox), np.array(gt_bbox))
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                
                # 判断是否为TP
                if best_iou >= iou_threshold:
                    tp[i] = 1
                    if best_gt_idx >= 0:
                        gt_matches[image_id][best_gt_idx]['matched'] = True
                else:
                    fp[i] = 1
            
            # 计算累积的TP和FP
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            # 计算精确率和召回率
            precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
            recalls = cum_tp / len(all_gts) if len(all_gts) > 0 else np.zeros_like(cum_tp)
            
            # 计算AP（PR曲线下面积）
            ap = calculate_ap(precisions, recalls)
            
            ap_results[iou_threshold][class_name] = {
                'ap': ap,
                'precisions': precisions.tolist(),
                'recalls': recalls.tolist(),
                'num_predictions': len(all_predictions),
                'num_gts': len(all_gts)
            }
        
        # 计算平均AP
        aps = [ap_results[iou_threshold][cls]['ap'] for cls in class_names if cls in ap_results[iou_threshold]]
        ap_results[iou_threshold]['average'] = {
            'ap': np.mean(aps) if aps else 0.0,
            'num_predictions': sum([ap_results[iou_threshold][cls]['num_predictions'] for cls in class_names if cls in ap_results[iou_threshold]]),
            'num_gts': sum([ap_results[iou_threshold][cls]['num_gts'] for cls in class_names if cls in ap_results[iou_threshold]])
        }
    
    return ap_results

def calculate_map(ap_results: Dict, iou_threshold: float = 0.5) -> float:
    """
    计算mAP (mean Average Precision)
    Args:
        ap_results: AP结果字典
        iou_threshold: IoU阈值
    Returns:
        mAP值
    """
    if iou_threshold not in ap_results:
        # 找到最接近的阈值
        available_thresholds = list(ap_results.keys())
        iou_threshold = min(available_thresholds, key=lambda x: abs(x - iou_threshold))
    
    aps = []
    for class_name, result in ap_results[iou_threshold].items():
        if class_name != 'average' and 'ap' in result:
            aps.append(result['ap'])
    
    return np.mean(aps) if aps else 0.0

def plot_pr_curves(ap_results: Dict, class_names: List[str], iou_threshold: float = 0.5, save_dir: str = None):
    """
    绘制PR曲线
    Args:
        ap_results: AP结果字典
        class_names: 类别名称列表
        iou_threshold: IoU阈值
        save_dir: 保存目录
    """
    if iou_threshold not in ap_results:
        print(f"警告: IoU阈值 {iou_threshold} 不在结果中，使用可用的阈值")
        iou_threshold = list(ap_results.keys())[0]
    
    plt.figure(figsize=(12, 10))
    
    # 绘制每个类别的PR曲线
    for class_name in class_names:
        if class_name in ap_results[iou_threshold]:
            result = ap_results[iou_threshold][class_name]
            precisions = result['precisions']
            recalls = result['recalls']
            ap = result['ap']
            
            if len(precisions) > 0 and len(recalls) > 0:
                # 确保recalls是单调递增的
                sorted_indices = np.argsort(recalls)
                sorted_recalls = np.array(recalls)[sorted_indices]
                sorted_precisions = np.array(precisions)[sorted_indices]
                
                plt.plot(sorted_recalls, sorted_precisions, '-', linewidth=2, 
                        label=f'{class_name} (AP={ap:.3f})')
    
    # 绘制平均PR曲线
    if 'average' in ap_results[iou_threshold]:
        result = ap_results[iou_threshold]['average']
        # 对于平均曲线，我们需要计算所有类别的平均PR曲线
        all_recalls = []
        all_precisions = []
        
        for class_name in class_names:
            if class_name in ap_results[iou_threshold]:
                result = ap_results[iou_threshold][class_name]
                if len(result['recalls']) > 0 and len(result['precisions']) > 0:
                    all_recalls.append(result['recalls'])
                    all_precisions.append(result['precisions'])
        
        if all_recalls and all_precisions:
            # 找到共同的recall点进行插值
            min_recall = 0
            max_recall = 1
            recall_points = np.linspace(min_recall, max_recall, 101)
            
            interp_precisions = []
            for recalls, precisions in zip(all_recalls, all_precisions):
                # 确保recalls是单调递增的
                sorted_indices = np.argsort(recalls)
                sorted_recalls = np.array(recalls)[sorted_indices]
                sorted_precisions = np.array(precisions)[sorted_indices]
                
                # 插值到共同的recall点
                interp_precision = np.interp(recall_points, sorted_recalls, sorted_precisions, 
                                           left=1.0, right=0.0)
                interp_precisions.append(interp_precision)
            
            # 计算平均精度
            if interp_precisions:
                mean_precision = np.mean(interp_precisions, axis=0)
                map_value = calculate_map(ap_results, iou_threshold)
                
                plt.plot(recall_points, mean_precision, 'k-', linewidth=4, 
                        label=f'Average (mAP={map_value:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves (IoU={iou_threshold})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'pr_curve_iou_{iou_threshold}.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curves(ap_results: Dict, class_names: List[str], iou_threshold: float = 0.5, save_dir: str = None):
    """
    绘制ROC曲线
    Args:
        ap_results: AP结果字典
        class_name: 类别名称列表
        iou_threshold: IoU阈值
        save_dir: 保存目录
    """
    if iou_threshold not in ap_results:
        print(f"警告: IoU阈值 {iou_threshold} 不在结果中，使用可用的阈值")
        iou_threshold = list(ap_results.keys())[0]
    
    plt.figure(figsize=(12, 10))
    
    # 绘制每个类别的ROC曲线
    for class_name in class_names:
        if class_name in ap_results[iou_threshold]:
            result = ap_results[iou_threshold][class_name]
            precisions = result['precisions']
            recalls = result['recalls']
            num_predictions = result['num_predictions']
            num_gts = result['num_gts']
            
            if len(precisions) > 0 and len(recalls) > 0 and num_predictions > 0 and num_gts > 0:
                # 计算FPR (False Positive Rate)
                # FPR = FP / (FP + TN)，但目标检测中TN难以定义
                # 这里使用近似：FPR = FP / (总预测数)
                # 或者使用：FPR = 1 - Precision
                
                # 方法1: 使用FPR = FP / (FP + TN) ≈ FP / (总预测数)
                # 但总预测数不是TN，这是一个近似
                fprs = []
                tprs = recalls  # TPR = Recall
                
                for i, precision in enumerate(precisions):
                    # 计算FP
                    fp_rate = (i + 1) * (1 - precision) / num_predictions if num_predictions > 0 else 0
                    fprs.append(fp_rate)
                
                # 确保FPR是单调递增的
                sorted_indices = np.argsort(fprs)
                sorted_fprs = np.array(fprs)[sorted_indices]
                sorted_tprs = np.array(tprs)[sorted_indices]
                
                # 计算AUC
                if len(sorted_fprs) > 1 and len(sorted_tprs) > 1:
                    roc_auc = auc(sorted_fprs, sorted_tprs)
                    plt.plot(sorted_fprs, sorted_tprs, '-', linewidth=2, 
                            label=f'{class_name} (AUC={roc_auc:.3f})')
    
    # 绘制平均ROC曲线
    if 'average' in ap_results[iou_threshold]:
        # 对于平均曲线，我们需要计算所有类别的平均ROC曲线
        all_fprs = []
        all_tprs = []
        
        for class_name in class_names:
            if class_name in ap_results[iou_threshold]:
                result = ap_results[iou_threshold][class_name]
                precisions = result['precisions']
                recalls = result['recalls']
                num_predictions = result['num_predictions']
                
                if len(precisions) > 0 and len(recalls) > 0 and num_predictions > 0:
                    fprs = []
                    for i, precision in enumerate(precisions):
                        fp_rate = (i + 1) * (1 - precision) / num_predictions
                        fprs.append(fp_rate)
                    
                    if len(fprs) > 0 and len(recalls) > 0:
                        # 插值到共同的FPR点
                        fpr_points = np.linspace(0, 1, 101)
                        interp_tpr = np.interp(fpr_points, fprs, recalls, left=0.0, right=1.0)
                        all_tprs.append(interp_tpr)
        
        if all_tprs:
            mean_tpr = np.mean(all_tprs, axis=0)
            mean_auc = auc(fpr_points, mean_tpr)
            
            plt.plot(fpr_points, mean_tpr, 'k-', linewidth=4, 
                    label=f'Average (AUC={mean_auc:.3f})')
    
    # 绘制对角线（随机分类器）
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'ROC Curves (IoU={iou_threshold})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'roc_curve_iou_{iou_threshold}.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_all_pr_roc_curves(ap_results: Dict, class_names: List[str], save_dir: str = None):
    """
    绘制所有IoU阈值下的PR和ROC曲线
    Args:
        ap_results: AP结果字典
        class_names: 类别名称列表
        save_dir: 保存目录
    """
    iou_thresholds = sorted(ap_results.keys())
    
    # 创建2xN的子图布局
    fig, axes = plt.subplots(2, len(iou_thresholds), figsize=(6*len(iou_thresholds), 10))
    
    if len(iou_thresholds) == 1:
        axes = np.array([axes]).T  # 确保axes是2D数组
    
    # 为每个IoU阈值绘制PR和ROC曲线
    for i, iou_threshold in enumerate(iou_thresholds):
        # PR曲线
        ax_pr = axes[0, i]
        
        # 绘制每个类别的PR曲线
        for class_name in class_names:
            if class_name in ap_results[iou_threshold]:
                result = ap_results[iou_threshold][class_name]
                precisions = result['precisions']
                recalls = result['recalls']
                ap = result['ap']
                
                if len(precisions) > 0 and len(recalls) > 0:
                    sorted_indices = np.argsort(recalls)
                    sorted_recalls = np.array(recalls)[sorted_indices]
                    sorted_precisions = np.array(precisions)[sorted_indices]
                    
                    ax_pr.plot(sorted_recalls, sorted_precisions, '-', linewidth=1, 
                              label=f'{class_name} (AP={ap:.3f})')
        
        # 绘制平均PR曲线
        if 'average' in ap_results[iou_threshold]:
            result = ap_results[iou_threshold]['average']
            map_value = calculate_map(ap_results, iou_threshold)
            
            # 计算所有类别的平均PR曲线
            all_recalls = []
            all_precisions = []
            
            for class_name in class_names:
                if class_name in ap_results[iou_threshold]:
                    result = ap_results[iou_threshold][class_name]
                    if len(result['recalls']) > 0 and len(result['precisions']) > 0:
                        all_recalls.append(result['recalls'])
                        all_precisions.append(result['precisions'])
            
            if all_recalls and all_precisions:
                recall_points = np.linspace(0, 1, 101)
                interp_precisions = []
                
                for recalls, precisions in zip(all_recalls, all_precisions):
                    sorted_indices = np.argsort(recalls)
                    sorted_recalls = np.array(recalls)[sorted_indices]
                    sorted_precisions = np.array(precisions)[sorted_indices]
                    
                    interp_precision = np.interp(recall_points, sorted_recalls, sorted_precisions, 
                                               left=1.0, right=0.0)
                    interp_precisions.append(interp_precision)
                
                if interp_precisions:
                    mean_precision = np.mean(interp_precisions, axis=0)
                    ax_pr.plot(recall_points, mean_precision, 'k-', linewidth=3, 
                              label=f'Average (mAP={map_value:.3f})')
        
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'PR Curve (IoU={iou_threshold})')
        ax_pr.grid(True, alpha=0.3)
        ax_pr.set_xlim([0, 1])
        ax_pr.set_ylim([0, 1])
        
        # ROC曲线
        ax_roc = axes[1, i]
        
        # 绘制每个类别的ROC曲线
        for class_name in class_names:
            if class_name in ap_results[iou_threshold]:
                result = ap_results[iou_threshold][class_name]
                precisions = result['precisions']
                recalls = result['recalls']
                num_predictions = result['num_predictions']
                
                if len(precisions) > 0 and len(recalls) > 0 and num_predictions > 0:
                    fprs = []
                    for j, precision in enumerate(precisions):
                        fp_rate = (j + 1) * (1 - precision) / num_predictions
                        fprs.append(fp_rate)
                    
                    if len(fprs) > 0:
                        sorted_indices = np.argsort(fprs)
                        sorted_fprs = np.array(fprs)[sorted_indices]
                        sorted_tprs = np.array(recalls)[sorted_indices]
                        
                        if len(sorted_fprs) > 1 and len(sorted_tprs) > 1:
                            roc_auc = auc(sorted_fprs, sorted_tprs)
                            ax_roc.plot(sorted_fprs, sorted_tprs, '-', linewidth=1, 
                                      label=f'{class_name} (AUC={roc_auc:.3f})')
        
        # 绘制平均ROC曲线
        if 'average' in ap_results[iou_threshold]:
            all_fprs = []
            all_tprs = []
            
            for class_name in class_names:
                if class_name in ap_results[iou_threshold]:
                    result = ap_results[iou_threshold][class_name]
                    precisions = result['precisions']
                    recalls = result['recalls']
                    num_predictions = result['num_predictions']
                    
                    if len(precisions) > 0 and len(recalls) > 0 and num_predictions > 0:
                        fprs = []
                        for j, precision in enumerate(precisions):
                            fp_rate = (j + 1) * (1 - precision) / num_predictions
                            fprs.append(fp_rate)
                        
                        if len(fprs) > 0:
                            fpr_points = np.linspace(0, 1, 101)
                            interp_tpr = np.interp(fpr_points, fprs, recalls, left=0.0, right=1.0)
                            all_tprs.append(interp_tpr)
            
            if all_tprs:
                mean_tpr = np.mean(all_tprs, axis=0)
                mean_auc = auc(fpr_points, mean_tpr)
                ax_roc.plot(fpr_points, mean_tpr, 'k-', linewidth=3, 
                           label=f'Average (AUC={mean_auc:.3f})')
        
        # 绘制对角线
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curve (IoU={iou_threshold})')
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_xlim([0, 1])
        ax_roc.set_ylim([0, 1])
    
    # 添加图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
               ncol=min(3, len(labels)), fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为图例留出空间
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'all_pr_roc_curves.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def print_detailed_metrics_per_class(results: Dict, class_names: List[str], ap_results: Dict = None):
    """
    打印详细评估指标（按类别）
    Args:
        results: 评估结果字典
        class_names: 类别名称列表
        ap_results: AP结果字典
    """
    print("=" * 120)
    print("多图片多类别目标检测评估结果")
    print("=" * 120)
    
    # 计算mAP
    map_results = calculate_map_per_class(results, class_names)
    
    # 打印每个类别的详细指标（以不同IoU阈值）
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for iou_threshold in iou_thresholds:
        print(f"\n各类别在IoU={iou_threshold}下的指标:")
        print("类别名称        | Precision | Recall   | F1 Score | Accuracy | TP  | FP  | FN  | mAP    | AP     ")
        print("-" * 120)
        
        for class_name in class_names + ['average']:
            if class_name in results and iou_threshold in results[class_name]:
                result = results[class_name][iou_threshold]
                map_value = map_results.get(class_name, 0)
                
                # 获取AP值（如果提供了ap_results）
                ap_value = 0.0
                if ap_results and iou_threshold in ap_results and class_name in ap_results[iou_threshold]:
                    ap_value = ap_results[iou_threshold][class_name].get('ap', 0.0)
                
                print(f"{class_name:<15} | {result['precision']:.4f}   | {result['recall']:.4f}   | "
                      f"{result['f1_score']:.4f}  | {result['accuracy']:.4f}  | {result['tp']:3d} | {result['fp']:3d} | {result['fn']:3d} | "
                      f"{map_value:.4f} | {ap_value:.4f}")
    
    # 打印AP和mAP总结
    if ap_results:
        print(f"\nAP和mAP总结:")
        print("IoU阈值 | mAP50   | AP50 (平均) | 备注")
        print("-" * 60)
        
        for iou_threshold in sorted(ap_results.keys()):
            map50 = calculate_map(ap_results, iou_threshold)
            ap_avg = ap_results[iou_threshold]['average']['ap'] if 'average' in ap_results[iou_threshold] else 0.0
            
            # 特别标注AP50
            note = " (AP50)" if abs(iou_threshold - 0.5) < 0.01 else ""
            print(f"{iou_threshold:.2f}    | {map50:.4f}  | {ap_avg:.4f}    | {note}")
    
    print("=" * 120)

def plot_metrics_per_class(results: Dict, class_names: List[str], ap_results: Dict = None, save_dir: str = None):
    """
    绘制每个类别的指标曲线
    Args:
        results: 评估结果字典
        class_names: 类别名称列表
        ap_results: AP结果字典
        save_dir: 保存目录
    """
    # 绘制每个类别的PR曲线
    plt.figure(figsize=(20, 10))
    
    iou_thresholds = sorted(results[class_names[0]].keys()) if class_names else []
    
    # PR曲线（基于固定IoU阈值）
    plt.subplot(2, 3, 1)
    for class_name in class_names:
        if class_name in results:
            precisions = [results[class_name][iou]['precision'] for iou in iou_thresholds]
            recalls = [results[class_name][iou]['recall'] for iou in iou_thresholds]
            plt.plot(recalls, precisions, 'o-', label=class_name, markersize=4)
    
    if 'average' in results:
        precisions = [results['average'][iou]['precision'] for iou in iou_thresholds]
        recalls = [results['average'][iou]['recall'] for iou in iou_thresholds]
        plt.plot(recalls, precisions, 'k-', linewidth=3, label='Average', markersize=6)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Fixed IoU)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 指标随IoU变化
    plt.subplot(2, 3, 2)
    if 'average' in results:
        precisions = [results['average'][iou]['precision'] for iou in iou_thresholds]
        recalls = [results['average'][iou]['recall'] for iou in iou_thresholds]
        f1_scores = [results['average'][iou]['f1_score'] for iou in iou_thresholds]
        
        plt.plot(iou_thresholds, precisions, 'b-o', label='Precision', markersize=4)
        plt.plot(iou_thresholds, recalls, 'r-o', label='Recall', markersize=4)
        plt.plot(iou_thresholds, f1_scores, 'g-o', label='F1 Score', markersize=4)
    
    plt.xlabel('IoU Threshold')
    plt.ylabel('Score')
    plt.title('Average Metrics vs IoU Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 每个类别的mAP柱状图
    plt.subplot(2, 3, 3)
    map_results = calculate_map_per_class(results, class_names)
    classes_to_plot = [cls for cls in class_names if cls in map_results]
    map_values = [map_results[cls] for cls in classes_to_plot]
    
    plt.bar(classes_to_plot, map_values)
    plt.xlabel('Class')
    plt.ylabel('mAP')
    plt.title('mAP per Class')
    plt.xticks(rotation=45)
    
    # 每个类别的AP柱状图（IoU=0.5）
    plt.subplot(2, 3, 4)
    if ap_results and 0.5 in ap_results:
        ap_values = []
        ap_classes = []
        for class_name in class_names:
            if class_name in ap_results[0.5]:
                ap_values.append(ap_results[0.5][class_name]['ap'])
                ap_classes.append(class_name)
        
        plt.bar(ap_classes, ap_values)
        plt.xlabel('Class')
        plt.ylabel('AP@0.5')
        plt.title('AP@0.5 per Class')
        plt.xticks(rotation=45)
    
    # AP随IoU阈值变化
    plt.subplot(2, 3, 5)
    if ap_results:
        for class_name in class_names[:5]:  # 只显示前5个类别，避免过于拥挤
            aps = []
            ious = []
            for iou_threshold in sorted(ap_results.keys()):
                if class_name in ap_results[iou_threshold]:
                    aps.append(ap_results[iou_threshold][class_name]['ap'])
                    ious.append(iou_threshold)
            
            if aps:
                plt.plot(ious, aps, 'o-', label=class_name, markersize=4)
        
        # 绘制平均AP
        avg_aps = []
        ious = []
        for iou_threshold in sorted(ap_results.keys()):
            if 'average' in ap_results[iou_threshold]:
                avg_aps.append(ap_results[iou_threshold]['average']['ap'])
                ious.append(iou_threshold)
        
        if avg_aps:
            plt.plot(ious, avg_aps, 'k-', linewidth=3, label='Average', markersize=6)
        
        plt.xlabel('IoU Threshold')
        plt.ylabel('AP')
        plt.title('AP vs IoU Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # mAP随IoU阈值变化
    plt.subplot(2, 3, 6)
    if ap_results:
        map_values = []
        iou_thresholds_sorted = sorted(ap_results.keys())
        for iou_threshold in iou_thresholds_sorted:
            map_val = calculate_map(ap_results, iou_threshold)
            map_values.append(map_val)
        
        plt.plot(iou_thresholds_sorted, map_values, 'ro-', linewidth=2, markersize=6)
        plt.xlabel('IoU Threshold')
        plt.ylabel('mAP')
        plt.title('mAP vs IoU Threshold')
        plt.grid(True, alpha=0.3)
        
        # 标记mAP50
        if 0.5 in ap_results:
            map50 = calculate_map(ap_results, 0.5)
            plt.annotate(f'mAP50: {map50:.4f}', (0.5, map50), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'metrics_per_class.png'), dpi=300, bbox_inches='tight')
    plt.show()

def read_pred_boxes_csv_multiclass(csv_path):
    """
    读取多类别预测框CSV文件
    格式: Image Index,X,Y,W,H,confidence,gt label
    """
    df = pd.read_csv(csv_path)
    all_pred_boxes = defaultdict(lambda: defaultdict(list))
    all_pred_boxes_with_conf = defaultdict(lambda: defaultdict(list))  # 新增：包含置信度的版本
    
    for _, row in df.iterrows():
        image_id = row['Image Index']
        x = row['X']
        y = row['Y']
        w = row['W']
        h = row['H']
        confidence = row['confidence']
        class_name = row['gt label']
        
        # 将(x, y, w, h)转换为(x1, y1, x2, y2)
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        
        all_pred_boxes[class_name][image_id].append([x1, y1, x2, y2])
        all_pred_boxes_with_conf[class_name][image_id].append([x1, y1, x2, y2, confidence])
    
    return all_pred_boxes, all_pred_boxes_with_conf

def read_gt_boxes_csv_multiclass(csv_path):
    """
    读取多类别真实框CSV文件
    格式: Image Index, Finding Label, Bbox [x, y, w, h]
    """
    df = pd.read_csv(csv_path)
    all_gt_boxes = defaultdict(lambda: defaultdict(list))
    disease_map = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    
    for _, row in df.iterrows():
        image_id = row['Image Index']
        class_name = row['Finding Label']
        class_index = disease_map.index(class_name)
        
        x, y, w, h = row[2:6]
        x = float(x)*448/1024
        y = float(y)*448/1024
        w = float(w)*448/1024
        h = float(h)*448/1024
        
        # 将(x, y, w, h)转换为(x1, y1, x2, y2)
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        
        all_gt_boxes[class_index][image_id].append([x1, y1, x2, y2])
    
    return all_gt_boxes

def save_results_to_json_per_class(results: Dict, class_names: List[str], filepath: str):
    """
    将多类别结果保存为JSON文件
    Args:
        results: 评估结果字典
        class_names: 类别名称列表
        filepath: 文件路径
    """
    serializable_results = {}
    
    for class_name in class_names + ['average']:
        if class_name in results:
            serializable_results[class_name] = {}
            for iou_threshold, result in results[class_name].items():
                serializable_results[class_name][str(iou_threshold)] = {
                    k: (float(v) if isinstance(v, (np.float32, np.float64)) else v)
                    for k, v in result.items()
                }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def save_ap_results_to_json(ap_results: Dict, filepath: str):
    """
    将AP结果保存为JSON文件
    Args:
        ap_results: AP结果字典
        filepath: 文件路径
    """
    serializable_results = {}
    
    for iou_threshold, class_results in ap_results.items():
        serializable_results[str(iou_threshold)] = {}
        for class_name, result in class_results.items():
            serializable_results[str(iou_threshold)][class_name] = {
                k: (float(v) if isinstance(v, (np.float32, np.float64)) else v)
                for k, v in result.items()
            }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

# 示例使用
def main_multiclass(all_pred_boxes, all_pred_boxes_with_conf, all_gt_boxes):
    # 获取所有类别名称
    all_classes = set(list(all_pred_boxes.keys()) + list(all_gt_boxes.keys()))
    class_names = sorted(list(all_classes))
    
    print(f"类别数量: {len(class_names)}")
    print(f"类别列表: {class_names}")
    
    # 统计每个类别的框数量
    print("\n每个类别的框数量统计:")
    print("类别名称        | 预测框数 | 真实框数")
    print("-" * 40)
    for class_name in class_names:
        pred_count = sum(len(boxes) for boxes in all_pred_boxes.get(class_name, {}).values())
        gt_count = sum(len(boxes) for boxes in all_gt_boxes.get(class_name, {}).values())
        print(f"{class_name:<15} | {pred_count:8d} | {gt_count:8d}")
    
    # 评估检测结果
    print("\n开始评估...")
    results = evaluate_detection_dataset_per_class(all_pred_boxes, all_gt_boxes, class_names)
    
    # 使用置信度计算AP和mAP
    print("\n计算AP和mAP...")
    ap_results = evaluate_detection_with_confidence(all_pred_boxes_with_conf, all_gt_boxes, class_names)
    
    # 计算mAP50
    map50 = calculate_map(ap_results, 0.5)
    map10 = calculate_map(ap_results, 0.1)
    
    # 打印指标
    print_detailed_metrics_per_class(results, class_names, ap_results)
    
    # 绘制PR和ROC曲线
    print("\n绘制PR和ROC曲线...")
    
    # 绘制所有IoU阈值下的PR和ROC曲线
    plot_all_pr_roc_curves(ap_results, class_names, '/nvme2/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/dino')
    
    # 单独绘制AP50的PR和ROC曲线
    plot_pr_curves(ap_results, class_names, 0.5, '/nvme2/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/dino')
    plot_roc_curves(ap_results, class_names, 0.5, '/nvme2/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/dino')
    
    # 绘制其他指标曲线
    print("\n绘制其他评估曲线...")
    plot_metrics_per_class(results, class_names, ap_results, '/nvme2/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/dino')
    
    # 计算并打印mAP
    map_results = calculate_map_per_class(results, class_names)
    print(f"\n各类别mAP0.05:0.5:")
    for class_name, map_value in map_results.items():
        print(f"  {class_name}: {map_value:.4f}")
    
    print(f"\nmAP50: {map50:.4f}")
    print(f"\nmAP10: {map10:.4f}")
    
    # 保存结果
    save_results_to_json_per_class(results, class_names, '/nvme2/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/dino/detection_results_multiclass.json')
    save_ap_results_to_json(ap_results, '/nvme2/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/dino/ap_results.json')
    print("\n结果已保存到指定目录")
    
    return results, ap_results, class_names

if __name__ == "__main__":
    # 读取多类别数据
    all_pred_boxes, all_pred_boxes_with_conf = read_pred_boxes_csv_multiclass('/nvme2/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/dino/gradcam_box.csv')
    all_gt_boxes = read_gt_boxes_csv_multiclass('/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/BBox_List_2017_onebox.csv')
    
    results, ap_results, class_names = main_multiclass(all_pred_boxes, all_pred_boxes_with_conf, all_gt_boxes)