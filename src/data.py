
import numpy as np
import pandas as pd
import os, time, logging, subprocess, platform
import random
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from src.utils import SCN, read_arrowhead_result

# load an HiC file with his resolution and return a matrix in numpy format
def load_hic(path, resolution):
	# Get the number of rows and columns of the matrix
	df = pd.read_csv(path, sep="\t",header=None, names=["i","j","score"])

	max_index = int(max(max(df.i)/resolution, max(df.j)/resolution))
	del df

	# Create square matrix
	matrix = np.zeros((max_index+1, max_index+1), dtype=int)

	# function that puts a value twice in the matrix
	def set_score(matrix, line):
		i,j,score = line.strip().split('\t')
		i = int(int(i)/resolution)
		j = int(int(j)/resolution)
		matrix[i,j] = int(float(score))
		matrix[j,i] = int(float(score))

	# fill the matrix
	f = open(path, 'r')
	lines = f.readlines()    
	for line in lines:
		set_score(matrix, line)
	return matrix

# preprocess all the Hic files contained in a folder (with the same resolution)
def preprocess_data(folder, resolution):
	start_time = time.time()
	logging.basicConfig(filename="data.log", level=logging.DEBUG)
	logging.info('===================================================================================')
	logging.info('\tPreprocessing of folder {} started...'.format(folder))
	for f in os.listdir(folder):
		if f.endswith('.RAWobserved'):
			m = load_hic(os.path.join(folder, f), resolution=resolution)
			# put the matrix in a numpy file
			np.save(os.path.join(folder, f.replace(".RAWobserved",".npy")), m)
			# put the matrix in a .txt (for TADtree)
			np.savetxt(os.path.join(folder, f.replace(".RAWobserved",".txt")), m, delimiter=' ', fmt='%d')
			logging.info('Preprocessing: file {} preprocessed'.format(f))
		else:
			logging.info('Preprocessing: file {} skipped'.format(f))
	logging.info("Preprocessing finished after {} seconds".format(time.time() - start_time))

# plot a contact map of an HiC file, possibility to zoom on a zone and to delimite it 
def plot_data(m, resolution, region=None, scale='log', tads=None):
	original_len = len(m)
	if scale == 'log':
		m = np.log10(m)
		m = SCN(m)
		# Vmax = m.max()/np.log10(len(m)/10)
		Vmax = m.max()
		Vmin = m.min()
		# TODO: find something for contrast diagonal / other
	if type(region) is tuple:
		# Subset of the file - zoom on a region
		if resolution is None:
			raise ValueError("Resolution must be specified for zoomed plots")
		# dezoom a bit to highlight the region
		region_length = (region[1]-region[0])/resolution
		dezoom = int(region_length/20)
		# TODO: Check
		start = max(int((region[0]/resolution)-dezoom),0)
		end = min(int((region[1]/resolution)+dezoom),original_len-1)
		m = m[start:end, start:end]
		# Vmax = m.max()/np.log10(len(m)/10)
	# else:
		# Vmax = m.max()/np.log10(len(m)/10)

	fig, ax = plt.subplots()
	shw = ax.imshow(m, cmap='OrRd', interpolation ='none', 
			  origin ='upper')
	
	if type(region) is tuple:
		start_idx = max(int(region[0]-dezoom),0)
	else:
		start_idx = 0
	xticks, _ = plt.xticks()
	xticks_cor = xticks[1:-1]
	yticks, _ = plt.yticks()
	yticks_cor = yticks[1:-1]
	plt.xticks(ticks=xticks_cor, labels=['{}'.format(int(((b+start_idx)*resolution)/1000000)) for b in xticks_cor])
	plt.yticks(ticks=yticks_cor, labels=['{}'.format(int(((b+start_idx)*resolution)/1000000)) for b in yticks_cor])
	bar = plt.colorbar(shw)
	bar.set_label('Scale')
	if tads is not None:
		for tad in tads:
			tad_length = (tad[1]-tad[0]) / resolution
			xy = (int(tad[0]/resolution), int(tad[0]/resolution))
			ax.add_patch(Rectangle(xy, tad_length, tad_length, fill=False, edgecolor='blue', linewidth=1))
	if region is not None:
		dezoom_left = min(start, dezoom)
		dezoom_right = min(original_len-end, dezoom)
		ax.add_patch(Rectangle((dezoom_left, dezoom_left), 
								len(m)-(dezoom_left+dezoom_right),
								len(m)-(dezoom_left+dezoom_right), 
								fill=False,
								edgecolor='black',
								linewidth=2))
	plt.show()

class Hicmat:
	def __init__(self, path, resolution, auto_filtering=True, cell_type=None):
		if path.endswith('.npy'):
			self.path = path
		else:
			self.path = path + '.npy'
		
		m = np.load(self.path)
		if m.shape[0] != m.shape[1]:
			raise ValueError('Matrix is not square')
		self.resolution = resolution
		self.original_matrix = np.array(m)
		self.filtered_coords = None
		self.reduced_matrix = None
		self.regions = None

		if auto_filtering:
			self.filter(threshold=1)

		self.set_cell_type(cell_type)

	def set_cell_type(self, cell_type):
		path_comps = self.path.split(os.sep)

		if 'GM12878' in path_comps:
			self.cell_type = 'GM12878'
		elif 'HMEC' in path_comps:
			self.cell_type = 'HMEC'
		elif 'HUVEC' in path_comps:
			self.cell_type = 'HUVEC'
		elif 'IMR90' in path_comps:
			self.cell_type = 'IMR90'
		elif 'NHEK' in path_comps:
			self.cell_type = 'NHEK'
		else:
			if cell_type is not None:
				self.cell_type = cell_type
			else:
				raise ValueError('HiC data: Cell Type neither known or specified')

	def filter(self, threshold = 0, min_length_region=5): # TODO: Discuss about min_length_region
		if self.filtered_coords is not None or self.reduced_matrix is not None:
			logging.info('Matrix already filtered')
			return
		sum_row_col = self.original_matrix.sum(axis=0) + self.original_matrix.sum(axis=1)
		self.filtered_coords = np.where(sum_row_col <= (threshold*(self.original_matrix.shape[0]+self.original_matrix.shape[1])) )[0]

		self.regions = []
		for i in range(len(self.filtered_coords)-1):
			# Save indexes of regions
			if self.filtered_coords[i+1] - self.filtered_coords[i] >= min_length_region:
				self.regions.append((self.filtered_coords[i], self.filtered_coords[i+1]))

		self.reduced_matrix = self.original_matrix.copy()
		self.reduced_matrix = np.delete(self.reduced_matrix, self.filtered_coords, axis=0)
		self.reduced_matrix = np.delete(self.reduced_matrix, self.filtered_coords, axis=1)

	def get_regions(self):
		if self.regions is None:
			raise ValueError('Matrix not filtered')
		return self.regions

	def get_folder(self):
		return os.path.dirname(self.path)

	def get_name(self):
		return os.path.basename(self.path)



def load_hic_groundtruth(data_path, resolution, arrowhead_folder=os.path.join('data', 'TADs'), threshold=1, cell_type=None, chr=None):
	"""
		Function to correctly load the HiC matrix and its corresponding Arrowhead results
	"""
	path_comps = data_path.split(os.sep)

	if 'GM12878' in path_comps or (cell_type is not None and cell_type == 'GM12878'):
		# GM12878
		if not chr:
			chr = path_comps[path_comps.index('GM12878')+2].split('_')[0][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_GM12878_primary+replicate_Arrowhead_domainlist.txt'), chromosome=chr)
	elif 'HMEC' in path_comps or (cell_type and cell_type == 'HMEC'):
		# HMEC
		if not chr:
			chr = path_comps[path_comps.index('HMEC')+2][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_HMEC_Arrowhead_domainlist.txt'), chromosome=chr)
	elif 'HUVEC' in path_comps or (cell_type is not None and cell_type == 'HUVEC'):
		# HUVEC
		if not chr:
			chr = path_comps[path_comps.index('HUVEC')+2][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_HUVEC_Arrowhead_domainlist.txt'), chromosome=chr)
	elif 'IMR90' in path_comps or (cell_type is not None and cell_type == 'IMR90'):
		# IMR90
		if not chr:
			chr = path_comps[path_comps.index('IMR90')+2][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_IMR90_Arrowhead_domainlist.txt'), chromosome=chr)
	elif 'NHEK' in path_comps or (cell_type is not None and cell_type == 'NHEK'):
		# NHEK
		if not chr:
			chr = path_comps[path_comps.index('NHEK')+2][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_NHEK_Arrowhead_domainlist.txt'), chromosome=chr)
	else:
		raise ValueError('HiC data: Cell Type and/or Chromosome neither known or specified')
	
	hic_mat = Hicmat(data_path, resolution, cell_type=cell_type)
	hic_mat.filter(threshold = threshold)

	return hic_mat, arrowhead_tads

class HiCDataset:
	def __init__(self, data_folder) -> None:
		self.data_folder = os.path.join(data_folder, 'HiC')
		self.development_set = None
		self.test_set = None 
		self.cell_types = ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'NHEK']
		self.resolutions = ['25kb_resolution_intrachromosomal', '100kb_resolution_intrachromosomal']
		self.data_dict = {res:{cell_type:[] for cell_type in self.cell_types} for res in self.resolutions}
		self.data_count = {res:{cell_type:0 for cell_type in self.cell_types} for res in self.resolutions}
		self.all_count = 0

	def preprocess_all(self):
		if platform.system() in ['Linux', 'Darwin']:
			bash_command_raw = 'find . -type f -name "*.RAWobserved" | wc -l'
			cmd_raw = subprocess.run([bash_command_raw], shell=True, capture_output=True, cwd=self.data_folder)
			bash_command_pre = 'find . -type f -name "*.npy" | wc -l'
			cmd_pre = subprocess.run([bash_command_pre], shell=True, capture_output=True, cwd=self.data_folder)
		elif platform.system() == 'Windows':
			powershell_command_raw = "(Get-ChildItem -recurse -filter *.RAWobserved | Measure-Object).Count"
			cmd_raw = subprocess.run(["powershell", "-Command", powershell_command_raw], shell=True, capture_output=True, cwd=self.data_folder)
			powershell_command_pre = "(Get-ChildItem -recurse -filter *.npy | Measure-Object).Count"
			cmd_pre = subprocess.run(["powershell", "-Command", powershell_command_pre], shell=True, capture_output=True, cwd=self.data_folder)
		else:
			raise ValueError('Unsupported platform {}'.format(platform.system()))
		raw_count = int(cmd_raw.stdout)
		pre_count = int(cmd_pre.stdout)

		if raw_count != pre_count:
			if pre_count != 0:
				raise ValueError('Preprocessed files already exist, but not all files have been processed - please manually check data folders')
			
			logging.info('Preprocessing all HiC data')
			for cell_type in self.cell_types:
				preprocess_data(os.path.join(self.data_folder, cell_type, '25kb_resolution_intrachromosomal'), 25000)
				for path in os.listdir(os.path.join(self.data_folder, cell_type, '25kb_resolution_intrachromosomal')):
					if os.path.isdir(os.path.join(self.data_folder, cell_type, '25kb_resolution_intrachromosomal', path)):
						preprocess_data(os.path.join(self.data_folder, cell_type, '25kb_resolution_intrachromosomal', path, 'MAPQGE30'), 25000)
			for cell_type in self.cell_types:
				preprocess_data(os.path.join(self.data_folder, cell_type, '100kb_resolution_intrachromosomal'), 100000)
				for path in os.listdir(os.path.join(self.data_folder, cell_type, '100kb_resolution_intrachromosomal')):
					if os.path.isdir(os.path.join(self.data_folder, cell_type, '100kb_resolution_intrachromosomal', path)):
						preprocess_data(os.path.join(self.data_folder, cell_type, '100kb_resolution_intrachromosomal', path, 'MAPQGE30'), 100000)
		else:
			logging.info('Preprocessing all HiC data already done')

	def build_data_dict(self):
		self.preprocess_all()

		for cell_type in self.cell_types:
			for resolution in self.resolutions:
				for path in os.listdir(os.path.join(self.data_folder, cell_type, resolution)):
					if path.endswith('.npy'):
						self.data_dict[resolution][cell_type].append(os.path.join(self.data_folder, cell_type, resolution, path))
						self.data_count[resolution][cell_type] += 1
						self.all_count += 1
					elif os.path.isdir(os.path.join(self.data_folder, cell_type, resolution, path)):
						if 'TADtree' in path or 'OnTAD' in path or 'TADbit' in path:
							continue
						for f in os.listdir(os.path.join(self.data_folder, cell_type, resolution, path, 'MAPQGE30')):
							if f.endswith('.npy'):
								self.data_dict[resolution][cell_type].append(os.path.join(self.data_folder, cell_type, resolution, path, 'MAPQGE30', f))
								self.data_count[resolution][cell_type] += 1
								self.all_count += 1

		if platform.system() in ['Linux', 'Darwin']:
			bash_command = 'find . -type f -name "*.RAWobserved" | wc -l'
			cmd = subprocess.run([bash_command], shell=True, capture_output=True, cwd=self.data_folder)
		elif platform.system() == 'Windows':
			powershell_command = "(Get-ChildItem -recurse -filter *.RAWobserved | Measure-Object).Count"
			cmd = subprocess.run(["powershell", "-Command", powershell_command], shell=True, capture_output=True, cwd=self.data_folder)
		else:
			raise ValueError('Unsupported platform {}'.format(platform.system()))
		cmd_count = int(cmd.stdout)
		if self.all_count != cmd_count:
			raise ValueError('Number of files considered for splits ({}) doesn\'t match the number of files in the dataset ({})'.format(self.all_count, cmd_count))


	def split(self, dev_ratio = 0.7, test_ratio=0.3, seed=123):
		assert dev_ratio + test_ratio == 1.0
		self.development_set = []
		self.test_set = []

		random.seed(seed) # Fix seed to assure reproducibility

		for resolution in self.resolutions:
			for cell_type in self.cell_types:
				random.shuffle(self.data_dict[resolution][cell_type])
				split_index = int(len(self.data_dict[resolution][cell_type]) * dev_ratio)
				self.development_set += self.data_dict[resolution][cell_type][:split_index]
				self.test_set += self.data_dict[resolution][cell_type][split_index:]

		return self.development_set, self.test_set
