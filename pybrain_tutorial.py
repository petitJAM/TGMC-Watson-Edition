from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer

ds = SupervisedDataSet(2, 1)

# for i in range(10):
ds.addSample((0,0), (0,))
ds.addSample((0,1), (1,))
ds.addSample((1,0), (1,))
ds.addSample((1,1), (0,))

net = buildNetwork(ds.indim, 4, ds.outdim, recurrent=True)
trainer = BackpropTrainer(net, ds, learningrate=0.01, momentum=0.99)
for epoch in range(0, 1000):
	trainer.train()

testdata = SupervisedDataSet(2, 1)
testdata.addSample((0,0), (0,))
testdata.addSample((0,1), (1,))
testdata.addSample((1,0), (1,))
testdata.addSample((1,1), (0,))

trainer.testOnData(testdata, verbose=True)