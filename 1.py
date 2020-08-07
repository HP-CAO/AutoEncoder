import tensorflow as tf

x0 = tf.Variable(3.0)
x1 = tf.Variable(0.0)

with tf.GradientTape() as tape:
  # Update x1 = x1 + x0.
    x1.assign_add(x0)
    print(x1)
  # The tape starts recording from x1.
    y = (x1+x0)**2   # y = (x1 + x0)**2
    print([var.name for var in tape.watched_variables()])
# This doesn't work.
print(tape.gradient(y, x0))   #dy/dx0 = 2*(x1 + x2)