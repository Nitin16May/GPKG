import inspect, torch
from torch_scatter import scatter

def scatter_(name, src, index, dim_size=None):
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
	return out[0] if isinstance(out, tuple) else out


class MessagePassing(torch.nn.Module):
	def __init__(self, aggr='add'):
		super(MessagePassing, self).__init__()
		print("MessagePassing")
		self.message_args = inspect.getfullargspec(self.message).args[1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
		self.update_args  = inspect.getfullargspec(self.update).args[2:]	# Same for update function starting from 3rd argument | first=self, second=out
		print(self.message_args)
		
	def propagate(self, aggr, edge_index, **kwargs):
		print("Propagate")
		assert aggr in ['add', 'mean', 'max']
		kwargs['edge_index'] = edge_index


		size = None
		message_args = []
		for arg in self.message_args:
			if arg[-2:] == '_i':					# If arguments ends with _i then include indic
				tmp  = kwargs[arg[:-2]]				# Take the front part of the variable | Mostly it will be 'x', 
				size = tmp.size(0)
				message_args.append(tmp[edge_index[0]])		# Lookup for head entities in edges
			elif arg[-2:] == '_j':
				tmp  = kwargs[arg[:-2]]				# tmp = kwargs['x']
				size = tmp.size(0)
				message_args.append(tmp[edge_index[1]])		# Lookup for tail entities in edges
			else:
				message_args.append(kwargs[arg])		# Take things from kwargs

		update_args = [kwargs[arg] for arg in self.update_args]		# Take update args from kwargs

		out = self.message(*message_args)				# Message function is called with the arguments
		out = scatter_(aggr, out, edge_index[0], dim_size=size)		# Aggregated neighbors for each vertex
		out = self.update(out, *update_args)

		return out

	def message(self, x_j):		 	# pragma: no cover
		print(x_j, "message")
		return x_j

	def update(self, aggr_out):  # pragma: no cover
		print(aggr_out, "update")
		return aggr_out
