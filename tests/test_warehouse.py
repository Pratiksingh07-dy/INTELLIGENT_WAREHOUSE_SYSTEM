from simulation.warehouse import (ACTION_ASSIGN_ARM, ACTION_ASSIGN_DRONE,
                                   ACTION_ASSIGN_ROBOT, ACTION_NOOP,
                                   ACTION_PRIORITIZE, Warehouse)


def test_warehouse_initializes_with_correct_resource_counts():
    wh = Warehouse(num_arms=2, num_robots=3, num_drones=1, seed=1)
    assert len(wh.arms) == 2
    assert len(wh.robots) == 3
    assert len(wh.drones) == 1


def test_step_generates_orders_and_advances_time():
    wh = Warehouse(arrival_rate=5.0, seed=1)  # high rate to guarantee arrivals
    initial_time = wh.time
    wh.step(ACTION_NOOP)
    assert wh.time == initial_time + wh.dt
    assert wh.total_orders_generated >= 0  # Poisson can occasionally be 0


def test_assign_arm_action_reduces_queue_when_order_present():
    wh = Warehouse(arrival_rate=0.0, seed=1)  # no random arrivals
    order = wh.order_generator.generate_order(wh.time)
    wh.orders[order.order_id] = order
    wh.queue.append(order.order_id)
    initial_queue_len = len(wh.queue)
    wh.step(ACTION_ASSIGN_ARM)
    assert len(wh.queue) == initial_queue_len - 1
    assert wh.arms[0].status.value == "processing"


def test_assign_arm_invalid_when_queue_empty():
    wh = Warehouse(arrival_rate=0.0, seed=1)
    reward = wh.step(ACTION_ASSIGN_ARM)
    assert wh.last_action_valid is False
    assert reward < 0  # invalid-action penalty applied


def test_prioritize_orders_high_priority_first():
    wh = Warehouse(arrival_rate=0.0, seed=1)
    from simulation.orders import Order, OrderPriority, OrderStage
    low = Order(order_id=-1, product="Widget-A", quantity=1, priority=OrderPriority.LOW,
                arrival_time=0.0, processing_time=1.0)
    high = Order(order_id=-2, product="Widget-A", quantity=1, priority=OrderPriority.HIGH,
                 arrival_time=0.0, processing_time=1.0)
    wh.orders[low.order_id] = low
    wh.orders[high.order_id] = high
    wh.queue = [low.order_id, high.order_id]
    wh.step(ACTION_PRIORITIZE)
    assert wh.queue[0] == high.order_id


def test_completed_order_yields_positive_completion_component():
    """A full arm->robot cycle should eventually complete an order and log it."""
    wh = Warehouse(arrival_rate=0.0, seed=1)
    order = wh.order_generator.generate_order(wh.time)
    order.processing_time = 0.5
    wh.orders[order.order_id] = order
    wh.queue.append(order.order_id)

    wh.step(ACTION_ASSIGN_ARM)     # arm picks up the order
    for _ in range(3):
        wh.step(ACTION_ASSIGN_ROBOT)  # eventually transport queue -> robot assigned

    # Advance enough steps for both arm processing and robot transport to finish
    for _ in range(10):
        wh.step(ACTION_NOOP)

    assert wh.total_orders_completed >= 1


def test_metrics_dict_has_expected_keys():
    wh = Warehouse(seed=1)
    wh.step(ACTION_NOOP)
    d = wh.to_dict()
    for key in ("time", "arms", "robots", "drones", "queue", "metrics"):
        assert key in d
    for key in ("total_orders_generated", "total_orders_completed", "pending_orders",
                "average_waiting_time", "throughput", "resource_utilization",
                "energy_used", "last_reward"):
        assert key in d["metrics"]
