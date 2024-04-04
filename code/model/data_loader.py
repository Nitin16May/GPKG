from model.tools import *
from collections import defaultdict as ddict
from torch.utils.data import Dataset

class TrainDataset(Dataset):
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		self.entities	= np.arange(self.p.num_ent, dtype=np.int32) 
		print("data loader starts here..")
		print(10)
		print(self.entities)
		# print(self.collate_fn())
		''' 
		This will make an array of numbers from 0 to num_ent-1.
		''' 
		# num_ent = number of entities in ent2id 

	def __len__(self):
		return len(self.triples)
	

	'''
		Things to look at  
		trp_label 
		collate_fn
	'''

	def __getitem__(self, idx):
		print(22)
		ele			= self.triples[idx]
		triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
		trp_label		= self.get_label(label)
		print(11)
		print(triple)
		print(label)
		print(sub_samp)
		print(trp_label)
		print("\n")
		'''
			trp_label is final smoothed label help in preventing overfitting.
		'''

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label, None, None
	'''
		
	'''
	def collate_fn(data):
		print(12)
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)

		if not data[0][2] is None:							# one_to_x
			neg_ent		= torch.stack([_[2] 	for _ in data], dim=0)
			sub_samp	= torch.cat([_[3] 	for _ in data], dim=0)
			return triple, trp_label, neg_ent, sub_samp
		else:
			return triple, trp_label
	
	''' 
	Here, the collate_fn function is used to stack the data in the data list into a single tensor.
	'''
	def get_label(self, label):
		print(13)
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0

		return torch.FloatTensor(y)
	
	''' 
	This function is used to create a one-hot encoding of the label.
	'''

class TestDataset(Dataset):
	def __init__(self, triples, params):
		print(14)
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		print(15)
		return len(self.triples)

	def __getitem__(self, idx):
		print(16)
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.int32(ele['label'])
		label		= self.get_label(label)

		return triple, label

	def collate_fn(data):
		print(17)
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label
	
	def get_label(self, label):
		print(18)
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)
