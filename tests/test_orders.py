from simulation.orders import OrderGenerator, OrderPriority, PRODUCT_CATALOG


def test_generate_order_has_valid_fields():
    gen = OrderGenerator(seed=1)
    order = gen.generate_order(current_time=0.0)
    assert order.order_id > 0
    assert order.product in PRODUCT_CATALOG
    assert 1 <= order.quantity <= 5
    assert order.priority in (OrderPriority.LOW, OrderPriority.NORMAL, OrderPriority.HIGH)
    assert order.processing_time > 0


def test_poisson_arrivals_nonnegative():
    gen = OrderGenerator(arrival_rate=1.5, seed=2)
    for _ in range(50):
        n = gen.poisson_arrivals()
        assert n >= 0


def test_arrival_rate_affects_average_arrivals():
    gen_low = OrderGenerator(arrival_rate=0.1, seed=3)
    gen_high = OrderGenerator(arrival_rate=3.0, seed=3)
    low_total = sum(gen_low.poisson_arrivals() for _ in range(300))
    high_total = sum(gen_high.poisson_arrivals() for _ in range(300))
    assert high_total > low_total


def test_generate_batch_returns_list():
    gen = OrderGenerator(arrival_rate=2.0, seed=4)
    batch = gen.generate_batch(current_time=5.0)
    assert isinstance(batch, list)
    for order in batch:
        assert order.arrival_time == 5.0


def test_set_arrival_rate():
    gen = OrderGenerator(arrival_rate=1.0, seed=5)
    gen.set_arrival_rate(2.5)
    assert gen.arrival_rate == 2.5
    gen.set_arrival_rate(-1.0)
    assert gen.arrival_rate == 0.0
