from model.tools import *
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from model.data_loader import *
from model.predict import *


class Main(object):

	def __init__(self, params):
		self.p = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model()
		self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)

	def load_data(self):
		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.upper, line.strip().split(' '))
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)
		print(1)
		print(ent_set)
		print(rel_set)
		print("\n")
		'''
		 First ordered entity set is created which have subject and object and an relation set defining relation between two
		 Then ent2id dictionary is created which has key as entity and is given an index starting from 0
		 same is done with rel2id 
		 now rel2id is updated with reverse relation. ( two make it biderectional )
		'''
		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		print(2)
		print(self.ent2id)
		print(self.rel2id)
		print("\n")
		'''
			here a reverse mapping is done where key is index and value is entity or relation respectively in id2ent and id2rel.
		'''
		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		print(3)
		print(self.id2ent)
		print(self.id2rel)
		print("\n")
		
		# Nmber of entity are directly equal to ent2id and num_rel is = to rel2id/2 because reverse relation are inserted 
		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		
		# Embedding dimension for entity and relation; height and width of reshaped matrix 
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		print(4)
		print(self.p.embed_dim,"\n")


		# The purpose of ddict(list) is to create a dictionary where the default value for any new key is an empty list ([]). 
		self.data	= ddict(list)
		sr2o		= ddict(set)  # Subject-Relation to Object

		print(type(sr2o))
		print(type(self.data))
		print("\n")
		'''
			Here we are first taking data from files and then getting the indexes of these relation
			instead of storing strings we take mapped integer value of sub obj and relation 
			this is then appended to respective split in data dictionary (self.data is dictionary)

			if the split was train we first add obj to set of sub and relation 
			and then we add sub to set of obj and reverse_relation
		'''

		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.upper, line.strip().split(' '))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
		self.data = dict(self.data)
		print(5)
		print(self.data)
		print(sr2o)
		print("\n")
		'''
			Here we are creating a dictionary of subject and relation as key and object as value
			the purpose of this line (self.sr2o = {k: list(v) for k, v in sr2o.items()}) is to create a new dictionary self.sr2o where the 
			values are lists (converted from sets) based on the contents of the original sr2o defaultdict.
		'''

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)
		
		# One more dictionary doing the same thing as above sub,rel as key and list as value
		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

		print(6)
		print(self.sr2o)
		print("\n")
		print(self.sr2o_all)
		print(type(self.sr2o))
		print(type(self.sr2o_all))
		print("\n")
	


		# Initialize a defaultdict 'self.triples' where the default value for any new key is an empty list.
		# Populate 'self.triples' with training triples, each represented by a dictionary with 'triple', 'label', and 'sub_samp' keys.
		self.triples = ddict(list)

		for (sub, rel), obj in self.sr2o.items():
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
		
		print(7)
		print(self.triples)
		print("\n")

		'''
			Here we are adding to triples dictionary with key as {test,valid} {head,tail} 
			if tail thenn  sub rel obj and label is sub and rel
			if head then obj rel_inv sub and label is obj and rel_inv
		'''

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)
		print(8)
		print(self.triples)
		# print(len(self.triples))
		# for key in self.triples:
		# 	print(f"The length of list at key '{key}' is {len(self.triples[key])}")
		print("\n")
		
		# batch_size default is 128.
		# num_worker default is 10.

		'''
			From here we will move to data_loader.py 
			because TrainDataset and TestDataset are defined here.
		'''
		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)
		
		print('om')

		self.data_iter = {
			'train'		:   get_data_loader(TrainDataset, 'train', 	self.p.batch_size),
			'valid_head'	:   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail'	:   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head'	:   get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail'	:   get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}
		print(9)
		# print(self.data_iter)
		print("\n")
		# print(20)
		# for key, dataloader in self.triples.items():
		# 	print(f"Data for '{key}':")
		# 	for batch in dataloader:
		# 		print(batch)
		self.chequer_perm	= self.get_chequer_perm()
		self.edge_index, self.edge_type = self.construct_adj()
	
	def construct_adj(self):
		edge_index, edge_type = [], []

		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		print(30)
		print(edge_index)
		print(edge_type)
		print("\n")
		# Adding inverse edges
		for sub, rel, obj in self.data['train']:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)

		print(31)
		print(edge_index)
		print(edge_type)
		print("\n")
		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)

		print(32)
		print(edge_index)
		print(edge_type)
		print("\n")
		return edge_index, edge_type
	
	def get_chequer_perm(self):
		ent_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)]) #self.p.perm = 1 
		rel_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])
		print("get_chequer_perm")
		comb_idx = []
		for k in range(self.p.perm):
			temp = []
			ent_idx, rel_idx = 0, 0

			for i in range(self.p.k_h):
				for j in range(self.p.k_w):
					if k % 2 == 0:
						if i % 2 == 0:
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
						else:
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
					else:
						if i % 2 == 0:
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
						else:
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;

			comb_idx.append(temp)
		# 	print(temp)
		# print("\n")
		# print(comb_idx)

		chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
		# print(chequer_perm)
		return chequer_perm


	def add_model(self):
		model = GPKG_PREDICT(self.edge_index, self.edge_type, self.chequer_perm, params=self.p)
		model.to(self.device)
		print(33)
		print(model)
		print("\n")
		return model

	def read_batch(self, batch, split):
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			print(34)
			print(triple)
			print(label)
			return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			print(35)
			print(triple)
			print(label)
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		print(36)
		print(state)
		print("\n")
		torch.save(state, save_path)

	def load_model(self, load_path):
		state				= torch.load(load_path)
		state_dict			= state['state_dict']
		self.best_val_mrr 		= state['best_val']['mrr']
		self.best_val 			= state['best_val']

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])
		print(37)
		print(state)
		print("\n")

	def evaluate(self, split, epoch=0):		
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}, hits@1 : {:.5}, hits@3 : {:.5}, hits@5 : {:.5}, hits@10 : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr'], results['hits@1'], results['hits@3'], results['hits@5'], results['hits@10']))
		print(39)
		print(results)
		print("\n")
		return results

	def predict(self, split='valid', mode='tail_batch'):
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred			= self.model.forward(sub, rel, None)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), torch.zeros_like(pred), pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))
		print(40)
		print(results)
		print("\n")
		return results

	def run_epoch(self, epoch):
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])

		for step, batch in enumerate(train_iter):
			self.optimizer.zero_grad()

			sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')

			pred	= self.model.forward(sub, rel, neg_ent)
			loss	= self.model.loss(pred, label, sub_samp)

			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}, \t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		print(41)
		print(loss)
		print("\n")
		return loss

	def fit(self):
		print(42)
		self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0.
		val_mrr = 0
		save_path = os.path.join('./model_saved', self.p.name)

		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		for epoch in range(self.p.max_epochs):
			train_loss	= self.run_epoch(epoch)
			val_results	= self.evaluate('valid', epoch)

			if val_results['mrr'] > self.best_val_mrr:
				self.best_val		= val_results
				self.best_val_mrr	= val_results['mrr']
				self.best_epoch		= epoch
				self.save_model(save_path)
			self.logger.info('[Epoch {}]:  Training Loss: {:.5},  Valid MRR: {:.5}, \n\n\n'.format(epoch, train_loss, self.best_val_mrr))

		print(43)
		# Restoring model corresponding to the best validation performance and evaluation on test data
		self.logger.info('Loading best model, evaluating on test data')
		self.load_model(save_path)		
		self.evaluate('test')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Dataset and Experiment name
	parser.add_argument('-data',           dest='dataset',         default='FB15k-237',            		help='Dataset to use for the experiment')
	parser.add_argument('-name',            			default='testrun_'+str(uuid.uuid4())[:8],	help='Name of the experiment')

	# Training parameters
	parser.add_argument('-gpu',		type=str,               default='0',					help='GPU to use, set -1 for CPU')
	parser.add_argument('-neg_num',        dest='neg_num',         default=1000,    	type=int,       	help='Number of negative samples to use for loss calculation')
	parser.add_argument('-batch',          dest='batch_size',      default=128,    	type=int,       	help='Batch size')
	parser.add_argument('-l2',		type=float,             default=0.0,					help='L2 regularization')
	parser.add_argument('-lr',		type=float,             default=0.001,					help='Learning Rate')
	parser.add_argument('-epoch',		dest='max_epochs', 	default=500,		type=int,  		help='Maximum number of epochs')
	parser.add_argument('-num_workers',	type=int,               default=10,                      		help='Maximum number of workers used in DataLoader')
	parser.add_argument('-seed',           dest='seed',            default=41504,   		type=int,       	help='Seed to reproduce results')
	parser.add_argument('-restore',   	dest='restore',       	action='store_true',            		help='Restore from the previously saved model')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')
	
	# Model parameters
	parser.add_argument('-lbl_smooth',     dest='lbl_smooth',	default=0.1,		type=float,		help='Label smoothing for true labels')
	parser.add_argument('-embed_dim',	type=int,              	default=None,                   		help='Embedding dimension for entity and relation, ignored if k_h and k_w are set')
	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')

	parser.add_argument('-bias',      	dest='bias',          	action='store_true',            		help='Whether to use bias in the model')
	parser.add_argument('-form',		type=str,               default='plain',            			help='The reshaping form to use')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   		type=int, 		help='Width of the reshaped matrix')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   		type=int, 		help='Height of the reshaped matrix')
	parser.add_argument('-num_filt',  	dest='num_filt',      	default=200,     	type=int,       	help='Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz',        	default=7,     		type=int,       	help='Kernel size to use')
	parser.add_argument('-perm',      	dest='perm',          	default=1,      	type=int,       	help='Number of Feature rearrangement to use')
	parser.add_argument('-hid_drop',  	dest='hid_drop',      	default=0.3,    	type=float,     	help='Dropout for Hidden layer')
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop',     	default=0.3,    	type=float,     	help='Dropout for Feature')
	parser.add_argument('-inp_drop',  	dest='inp_drop',      	default=0.1,    	type=float,     	help='Dropout for Input layer')

	# Logging parameters
	parser.add_argument('-logdir',    	dest='log_dir',       	default='./log/',               		help='Log directory')
	parser.add_argument('-config',    	dest='config_dir',    	default='./config/',            		help='Config directory')
	

	args = parser.parse_args()
	
	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Main(args)
	model.fit()
