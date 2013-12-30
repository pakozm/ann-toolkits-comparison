-- un generador de valores aleatorios... y otros parametros
bunch_size     = tonumber(arg[1]) or 64
semilla        = 1234
weights_random = random(semilla)
description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
inf            = -6
sup            =  6
shuffle_random = random(5678)
learning_rate  = 0.01
momentum       = 0.00
weight_decay   = 0.0 -- 1e-05
max_epochs     = 10

--------------------------------------------------------------

m1 = ImageIO.read(string.get_path(arg[0]) .. "digits.png"):to_grayscale():invert_colors():matrix()
train_input = dataset.matrix(m1,
			     {
			       patternSize = {16,16},
			       offset      = {0,0},
			       numSteps    = {80,10},
			       stepSize    = {16,16},
			       orderStep   = {1,0}
			     })

val_input  = dataset.matrix(m1,
			    {
			      patternSize = {16,16},
			      offset      = {1280,0},
			      numSteps    = {20,10},
			      stepSize    = {16,16},
			      orderStep   = {1,0}
			    })
-- una matriz pequenya la podemos cargar directamente
m2 = matrix(10,{1,0,0,0,0,0,0,0,0,0})

-- ojito con este dataset, fijaros que usa una matriz de dim 1 y talla
-- 10 PERO avanza con valor -1 y la considera CIRCULAR en su unica
-- dimension

train_output = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {800},
				stepSize    = {-1},
				circular    = {true}
			      })

val_output   = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {200},
				stepSize    = {-1},
				circular    = {true}
			      })

thenet = ann.mlp.all_all.generate(description)
trainer = trainable.supervised_trainer(thenet,
				       ann.loss.multi_class_cross_entropy(),
				       bunch_size)
trainer:build()

trainer:set_option("learning_rate", learning_rate)
trainer:set_option("momentum",      momentum)
trainer:set_option("weight_decay",  weight_decay)
-- bias has weight_decay of ZERO
trainer:set_layerwise_option("b.", "weight_decay", 0)

trainer:randomize_weights{
  random      = weights_random,
  inf         = inf,
  sup         = sup,
  use_fanin   = true,
}

-- datos para entrenar
datosentrenar = {
  input_dataset  = train_input,
  output_dataset = train_output,
  shuffle        = shuffle_random,
}

datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_output,
}

errorval = trainer:validate_dataset(datosvalidar)
-- print("# Initial validation error:", errorval)

clock = util.stopwatch()
clock:go()

-- print("Epoch Training  Validation")
train_func = trainable.train_holdout_validation{ max_epochs = max_epochs }
-- training loop
while train_func:execute(function()
			   local tr = trainer:train_dataset(datosentrenar)
			   local va = trainer:validate_dataset(datosvalidar)
			   return trainer,tr,va
			 end) do
  print(train_func:get_state_string())
  -- if best_epoch == epoch then train_func:save("wop.lua", "binary", {shuffle=datosentrenar.shuffle}) end
end
totalepochs = train_func:get_state_table().current_epoch

clock:stop()
cpu,wall = clock:read()
printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/totalepochs)
printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/totalepochs)
-- print("Test passed! OK!")
