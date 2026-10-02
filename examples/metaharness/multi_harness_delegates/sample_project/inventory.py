"""A tiny stock-keeping module for a corner shop."""


def restock(stock, item, quantity):
    """Add ``quantity`` of ``item`` to the stock and return the new count."""
    stock[item] = stock.get(item, 0) + quantity
    return stock[item]


def sell(stock, item, quantity):
    """Remove ``quantity`` of ``item``; refuse to sell more than is in stock."""
    if stock[item] < quantity:
        raise ValueError(f"only {stock[item]} {item} left")
    stock[item] -= quantity
    return stock[item]


def low_stock(stock, threshold=5):
    """Items at or below ``threshold``, most urgent first."""
    return sorted(
        [item for item, count in stock.items() if count < threshold],
        key=lambda item: stock[item],
    )


def order_total(prices, basket, discount=0):
    """Total price of a basket after a percentage ``discount`` (0 to 100)."""
    total = sum(prices[item] * quantity for item, quantity in basket.items())
    return round(total - total * discount, 2)
