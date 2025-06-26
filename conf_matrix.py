import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from get_labels import get_defect_type

def get_confusion_matrix(y, y_pred):
	cm = confusion_matrix(y, y_pred)
	return cm

def get_accuracy(cm):
	total_correct = np.trace(cm)
	total_samples = np.sum(cm)
	accuracy = total_correct / total_samples
	return accuracy

def plot_confusion_matrix(cm, plot_options={'plot_title':''}, save_opt_dict={'ifSave':False, 'path':'', 'file_name':''}):

	ifSave = save_opt_dict['ifSave']
	path = save_opt_dict['path']
	file_name = save_opt_dict['file_name']

	plot_title = plot_options['plot_title']

	fontsize_sec = 14
	fontsize_title = fontsize_sec+12
	fontsize_label = fontsize_sec+6
	fontsize_legend = fontsize_sec+6
	fontsize_plot = fontsize_sec+6

	fig, ax = plt.subplots(1,1, figsize=(10,10))

	display_labels = [get_defect_type(x) for x in range(4)]

	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
	disp.plot(cmap='Blues', values_format='d', ax=ax)
	for text_array in disp.text_:
		for text in text_array:
			text.set_fontsize(fontsize_plot)

	ax.set_title(plot_title, fontsize=fontsize_title, fontweight='bold')
	ax.tick_params(axis='both', labelsize=fontsize_label)
	ax.set_xlabel('Prediction', fontsize=fontsize_label+4, fontweight='bold')
	ax.set_ylabel('Ground truth', fontsize=fontsize_label+4, fontweight='bold')

	cbar = ax.images[0].colorbar
	cbar.ax.tick_params(labelsize=fontsize_legend)
	coordinate_value = 0.05
	cbar.ax.set_position([ax.get_position().x1 + coordinate_value, ax.get_position().y0, 
						  coordinate_value, ax.get_position().height])
	if ifSave:
		full_path = f'{path}/{file_name}'
		plt.savefig(full_path, bbox_inches='tight')