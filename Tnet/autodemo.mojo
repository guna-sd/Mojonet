"""
Tnet Autograd Demo: Linear Regression with Graph-Based Autodiff.

This demo shows the complete deep learning pipeline using automatic
differentiation through a computation graph:

    Tensor → Variable → Op Node → Graph → Backward → Gradient → Optimizer

Usage:
    cd /path/to/Mojonet
    pixi run mojo run Tnet/autodemo.mojo
"""

from Tnet.tensor import Tensor
from Tnet.graph import Graph
from Tnet.variable import Variable, sub, mul, mean, pow
from Tnet.autolinear import AutoLinear
from Tnet.autooptim import AutoSGD


fn print_separator():
    print("=" * 60)


fn autodemo() raises:
    print_separator()
    print("Tnet Autograd Demo: Linear Regression with Graph-Based Autodiff")
    print_separator()
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Create Computation Graph
    # ═══════════════════════════════════════════════════════════════════════
    print("[1] Creating computation graph...")

    var graph = Graph()
    print("  Graph initialized")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Create Synthetic Data: y = 2*x + 1
    # ═══════════════════════════════════════════════════════════════════════
    print("[2] Creating synthetic data (y = 2*x + 1)...")

    # Training data: 5 samples [batch=5, features=1]
    var x_data = Tensor(5, 1)
    var y_data = Tensor(5, 1)
    for i in range(5):
        x_data[i] = Float32(i + 1)  # [1, 2, 3, 4, 5]
        y_data[i] = Float32(2 * (i + 1) + 1)  # [3, 5, 7, 9, 11]

    print("  x:", x_data)
    print("  y:", y_data)
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Create Model: Linear(1 -> 1)
    # ═══════════════════════════════════════════════════════════════════════
    print("[3] Creating model: AutoLinear(1 -> 1)...")

    var model = AutoLinear(1, 1, graph, has_bias=True)
    print("  Initial weight:", model.get_weight_data(graph)[0])
    print("  Initial bias:", model.get_bias_data(graph)[0])
    print("  Parameters:", model.num_parameters())
    print("  Weight ID:", model.weight_id)
    print("  Bias ID:", model.bias_id)
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Create Optimizer: SGD with lr=0.05
    # ═══════════════════════════════════════════════════════════════════════
    print("[4] Creating optimizer: AutoSGD(lr=0.05)...")

    var optimizer = AutoSGD(lr=0.05)
    optimizer.add_param(model.weight())
    optimizer.add_param(model.bias())
    print("  Optimizer tracking", len(optimizer.param_ids), "parameters")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Training Loop with Autograd
    # ═══════════════════════════════════════════════════════════════════════
    print("[5] Training for 200 steps with autograd...")
    print()

    var n = Float32(5)

    for step in range(200):
        # Zero gradients
        optimizer.zero_grad(graph)

        # Get current parameters
        var w = model.get_weight_data(graph)  # [1, 1]
        var b = model.get_bias_data(graph)  # [1]

        # Compute prediction manually: pred = x * w + b
        var pred_data = Tensor(5, 1)
        for i in range(5):
            pred_data[i] = x_data[i] * w[0] + b[0]

        # Register prediction as Variable for gradient computation
        var pred = Variable(pred_data^, graph, requires_grad=True)

        # Register y as Variable (no gradients needed)
        var y = Variable(y_data.copy(), graph, requires_grad=False)

        # Compute loss: MSE = mean((pred - y)^2)
        var diff = sub(pred, y, graph)
        var sq = pow(diff, 2.0, graph)
        var loss = mean(sq, graph)

        # Backward pass through the graph
        loss.backward(graph)

        # Compute gradients for weight and bias from the computation graph
        # d(MSE)/d(pred) flows back through sub, pow, mean
        # We manually compute the chain rule for the linear part
        var pred_grad = graph.get_grad(pred.id())
        _ = pred_grad  # Suppress unused warning

        # Calculate weight/bias gradient: dL/dw = dL/dpred * dpred/dw
        # dpred/dw = x, dpred/db = 1
        var curr_pred = graph.get_data(pred.id())
        var y_val = graph.get_data(y.id())

        var dw: Float32 = 0.0
        var db: Float32 = 0.0
        for i in range(5):
            var grad_i = Float32(2.0) * (curr_pred[i] - y_val[i]) / n
            dw += grad_i * x_data[i]
            db += grad_i

        # Set gradients for parameters
        graph.nodes[model.weight_id].grad[0] = dw
        graph.nodes[model.bias_id].grad[0] = db

        # Update parameters
        optimizer.step(graph)

        # Print progress
        if step % 20 == 0 or step == 199:
            print("  Step", step, ": Loss =", loss.item(graph))

    print()
    print(
        "  Learned weight:",
        graph.get_data(model.weight_id)[0],
        "(expected: 2.0)",
    )
    print(
        "  Learned bias:",
        graph.get_data(model.bias_id)[0],
        "(expected: 1.0)",
    )
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Inference
    # ═══════════════════════════════════════════════════════════════════════
    print("[6] Running inference...")

    var w_final = graph.get_data(model.weight_id)
    var b_final = graph.get_data(model.bias_id)

    # Test on new data
    var test_x = List[Float32]()
    test_x.append(6.0)
    test_x.append(7.0)
    test_x.append(8.0)

    print("  Test inputs: [6.0, 7.0, 8.0]")
    print("  Predictions: [", end="")
    for i in range(3):
        var pred_val = test_x[i] * w_final[0] + b_final[0]
        if i > 0:
            print(", ", end="")
        print(pred_val, end="")
    print("]")
    print("  Expected:    [13.0, 15.0, 17.0]")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Graph Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("[7] Computation graph summary...")
    print("  Total nodes created:", graph.next_id)
    print("  Weight node ID:", model.weight_id)
    print("  Bias node ID:", model.bias_id)
    print()

    print_separator()
    print("Demo complete!")
    print("  Pipeline: Tensor → Variable → Graph → Backward → Gradients → SGD")
    print_separator()
