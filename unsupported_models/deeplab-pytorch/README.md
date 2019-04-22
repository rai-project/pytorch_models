Reference: https://github.com/kazuto1011/deeplab-pytorch

Errors: 
```
	python convert_python_free.py single --config-path ./configs/cocostuff164k.yaml --model-path ./pretrain/deeplabv2_resnet101_msc-cocostuff164k-100000.pth --image-path ./input.jpg
	Mode: single
	convert_python_free.py:142: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
	  CONFIG = Dict(yaml.load(config_path))
	Device: GeForce RTX 2070
	torch.Size([1, 3, 385, 513])
	Traceback (most recent call last):
	  File "convert_python_free.py", line 273, in <module>
	    main()
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 764, in __call__
	    return self.main(*args, **kwargs)
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 717, in main
	    rv = self.invoke(ctx)
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 1137, in invoke
	    return _process_result(sub_ctx.command.invoke(sub_ctx))
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 956, in invoke
	    return ctx.invoke(self.callback, **ctx.params)
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 555, in invoke
	    return callback(*args, **kwargs)
	  File "convert_python_free.py", line 162, in single
	    traced_script_module = torch.jit.trace(model, torch.randn(3,300,300).to(device))
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/jit/__init__.py", line 636, in trace
	    var_lookup_fn, _force_outplace)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 487, in __call__
	    result = self._slow_forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in _slow_forward
	    result = self.forward(*input, **kwargs)
	  File "/home/jasonwu/deeplab-pytorch/libs/models/msc.py", line 28, in forward
	    logits = self.base(x)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 487, in __call__
	    result = self._slow_forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in _slow_forward
	    result = self.forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/container.py", line 92, in forward
	    input = module(input)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 487, in __call__
	    result = self._slow_forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in _slow_forward
	    result = self.forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/container.py", line 92, in forward
	    input = module(input)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 487, in __call__
	    result = self._slow_forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in _slow_forward
	    result = self.forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/container.py", line 92, in forward
	    input = module(input)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 487, in __call__
	    result = self._slow_forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in _slow_forward
	    result = self.forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/conv.py", line 320, in forward
	    self.padding, self.dilation, self.groups)
	RuntimeError: Expected 4-dimensional input for 4-dimensional weight [64, 3, 7, 7], but got 3-dimensional input of size [3, 300, 300] instead
	(deeplab-pytorch) jasonwu@jasonwu-AX370M-DS3H:~/deeplab-pytorch$ python convert_python_free.py single --config-path ./configs/cocostuff164k.yaml --model-path ./pretrain/deeplabv2_resnet101_msc-cocostuff164k-100000.pth --image-path ./input.jpg
	Mode: single
	convert_python_free.py:142: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
	  CONFIG = Dict(yaml.load(config_path))
	Device: GeForce RTX 2070
	torch.Size([1, 3, 385, 513])
	Traceback (most recent call last):
	  File "convert_python_free.py", line 273, in <module>
	    main()
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 764, in __call__
	    return self.main(*args, **kwargs)
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 717, in main
	    rv = self.invoke(ctx)
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 1137, in invoke
	    return _process_result(sub_ctx.command.invoke(sub_ctx))
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 956, in invoke
	    return ctx.invoke(self.callback, **ctx.params)
	  File "/home/jasonwu/anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/click/core.py", line 555, in invoke
	    return callback(*args, **kwargs)
	  File "convert_python_free.py", line 162, in single
	    traced_script_module = torch.jit.trace(model, torch.randn(1,3,300,300).to(device))
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/jit/__init__.py", line 636, in trace
	    var_lookup_fn, _force_outplace)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 487, in __call__
	    result = self._slow_forward(*input, **kwargs)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in _slow_forward
	    result = self.forward(*input, **kwargs)
	  File "/home/jasonwu/deeplab-pytorch/libs/models/msc.py", line 37, in forward
	    h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
	  File "/home/jasonwu/.local/lib/python3.6/site-packages/torch/nn/functional.py", line 2447, in interpolate
	    return torch._C._nn.upsample_bilinear2d(input, _output_size(2), align_corners)
	RuntimeError: invalid argument 2: input and output sizes should be greater than 0, but got input (H: 300, W: 300) output (H: 0, W: 0) at /pytorch/aten/src/THCUNN/generic/SpatialUpSamplingBilinear.cu:17
```