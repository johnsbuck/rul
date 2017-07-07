import tensorflow as tf

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------

def rul_sign(y, model):
    """Used for obtaining the sign of y - model.

    Args:
        y (tf.Variables): Expected values
        model (tf.Variables): Actual values

    Returns: 1 if y - model > 0, 1/2 if y - model = 0, and 0 if y - model < 0

    """
    return tf.divide(tf.add(tf.sign(tf.subtract(y, model)), 1), 2)

# ------------------------------------------------
# REGRESSION
# ------------------------------------------------

def rul_left(y, model):
    """Used to equate score when (y - model) is negative.

    Args:
        y(tf.Variables): Expected values
        model(tf.Variables): Actual values

    Returns:

    """
    return tf.subtract(tf.reduce_sum(tf.exp(tf.divide(tf.subtract(model, y), -13))), 1)


def rul_right(y, model):
    """Used to equate score when (y - model) is positive.

    Args:
        y(tf.Variables): Expected values
        model(tf.Variables): Actual values

    Returns:

    """
    return tf.subtract(tf.reduce_sum(tf.exp(tf.divide(tf.subtract(model, y), 10))), 1)


def rul(y, model):
    """Score function for RUL Estimation.

    Args:
        y(tf.Variables): Expected values
        model(tf.Variables): Actual values

    Returns:

    """
    return tf.reduce_sum(
            tf.add(tf.multiply(rul_sign(y, model), rul_left(y, model)),
               tf.multiply(rul_sign(-y, -model), rul_right(y, model))))

# ------------------------------------------------
# LOGISTIC REGRESSION
# ------------------------------------------------

def mse(y, model, batch_size):
    """Mean-Squared Error

    Args:
        y (tf.Variables): Expected values
        model (tf.Variables): Actual values
        batch_size (int): Number of y and model values

    Returns:

    """
    return tf.divide(tf.reduce_sum(tf.square(tf.subtract(y, model))), batch_size)


def mae(y, model, batch_size):
    return tf.divide(tf.reduce_sum(tf.abs(tf.subtract(y, model))), batch_size)

# ------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------

def softmax_cross_entropy(y, model, batch_size):
    """Soft-Max Cross Entropy

    Args:
        y (tf.Variables): Expected values
        model (tf.Variables): Actual values
        batch_size (int): Number of y and model values

    Returns:

    """
    return tf.divide(tf.softmax_cross_entropy_with_logits(y, model), batch_size)
