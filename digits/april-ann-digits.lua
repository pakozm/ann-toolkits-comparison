-- un generador de valores aleatorios... y otros parametros
bunch_size     = tonumber(arg[1]) or 64
semilla        = 1234
weights_random = random(semilla)
description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
inf            = -6
sup            =  6
shuffle_random = random(5678)
learning_rate  = 0.01
max_epochs     = 40

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
  replacement    = train_input:numPatterns(),
}

datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_output,
}

errorval = trainer:validate_dataset(datosvalidar)
-- print("# Initial validation error:", errorval)
collectgarbage("collect")

clock = util.stopwatch()
clock:go()

-- training loop
for i=1,max_epochs do
  collectgarbage("collect")
  local tr = trainer:train_dataset(datosentrenar)
  local va = trainer:validate_dataset(datosvalidar)
  printf("%4d  %.6f  %.6f\n", i, tr, va)
end

clock:stop()
cpu,wall = clock:read()
printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
-- print("Test passed! OK!")
