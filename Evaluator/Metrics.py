# Code inspired by https://github.com/ubicomplab/rPPG-Toolbox 
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from Evaluator.post_process import *


def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))
def Scatterplot(path, x,y):
    rmse = np.sqrt(np.mean(np.square(x - y)))

    # Calculate error bars
    errors = np.abs(x - y)
    # Plot scatter with error bars
    fig, ax = plt.subplots()
    ax.scatter(x, y, marker='^', c='r', alpha=0.5, label='Predicted')  # Red triangles for predicted
    ax.scatter(x, x, marker='o', c='b', alpha=0.5, label='Ground truth')  # Blue circles for ground truth
    ax.errorbar(x, y, xerr=0, yerr=errors, fmt='none', ecolor='gray', alpha=0.5)

    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    # Set axis labels and title
    ax.set_xlabel('Ground truth heart rate')
    ax.set_ylabel('Predicted heart rate')
    ax.set_title('Scatter plot with error bars (RMSE = {:.2f})'.format(rmse))

    # Add legend
    ax.legend()
    plt.savefig(path)

def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    for index in predictions.keys():
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")
        gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
        gt_hr_peak, pred_hr_peak = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)
    predict_hr_peak_all = np.array(predict_hr_peak_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    print("---------------------------------------------------------------------------------------------------")
    print("Overall predicted :: ", np.mean(predict_hr_peak_all), "Overall Real :: ", np.mean(gt_hr_peak_all) )
    print("---------------------------------------------------------------------------------------------------")
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            #
            # Save the plot to a file
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
                x = gt_hr_fft_all  # Ground truth heart rate
                y = predict_hr_fft_all  # Predicted heart rate
                Scatterplot("/notebooks/HR-VViT-Contactless-Heart-Rate-Estimation-Based-on-VViTs/Exp1/Scatter_RMSE_FFT.png",x,y)
                

            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
                x = gt_hr_peak_all  # Ground truth heart rate
                y = predict_hr_peak_all  # Predicted heart rate
                Scatterplot("/notebooks/HR-VViT-Contactless-Heart-Rate-Estimation-Based-on-VViTs/Exp1/Scatter_RMSE_Peak.png",x,y)
                
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                Pearson_FFT_value = Pearson_FFT[0][1]
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT_value))
                fig, ax = plt.subplots()
                im = ax.imshow(Pearson_FFT, cmap='coolwarm')
                ax.set_xticks(np.arange(2))
                ax.set_yticks(np.arange(2))
                ax.set_xticklabels(['Predicted (FFT)', 'Ground Truth (FFT)'])
                ax.set_yticklabels(['Predicted (FFT)', 'Ground Truth (FFT)'])
                plt.colorbar(im)
                ax.set_title('Pearson Correlation Matrix (Pearson = {:.4f})'.format(Pearson_FFT_value))
                plt.savefig("/notebooks/HR-VViT-Contactless-Heart-Rate-Estimation-Based-on-VViTs/Exp1/Pearson_FFT.png")
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                Pearson_PEAK_value = Pearson_PEAK[0][1]
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK_value))
                fig, ax = plt.subplots()
                im = ax.imshow(Pearson_PEAK, cmap='coolwarm')
                ax.set_xticks(np.arange(2))
                ax.set_yticks(np.arange(2))
                ax.set_xticklabels(['Predicted (PEAK)', 'Ground Truth (PEAK)'])
                ax.set_yticklabels(['Predicted (PEAK)', 'Ground Truth (PEAK)'])
                plt.colorbar(im)
                ax.set_title('Pearson Correlation Matrix (Pearson = {:.4f})'.format(Pearson_PEAK_value))
                plt.savefig("/notebooks/HR-VViT-Contactless-Heart-Rate-Estimation-Based-on-VViTs/Exp1/Pearson_Peak.png")
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        else:
            raise ValueError("Wrong Test Metric Type")

def predict_calculate_metrics(predictions_batch):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.) using FFT."""
    predict_hr_fft_all = list()
    prediction = predictions_batch
    diff_flag_test = True
    pred_hr_fft = calculate_prediction(
        prediction, diff_flag=diff_flag_test, fs=30, hr_method='FFT')
    #predict_hr_fft_all.append(pred_hr_fft)
    pred_hr_peak = calculate_prediction(
        prediction, diff_flag=diff_flag_test, fs=30, hr_method='Peak')
    #predict_hr_fft_all.append(pred_hr_fft)    
    predict_hr_fft_all = np.array(pred_hr_fft)
    print("Predicted Heart Rate (FFT):", np.mean(predict_hr_fft_all))
    predict_hr_peak_all = np.array(pred_hr_peak)
    print("Predicted Heart Rate (FFT):", np.mean(predict_hr_peak_all))
    return predict_hr_fft_all