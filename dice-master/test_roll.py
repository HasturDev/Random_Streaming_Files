from app import roll

def test_roll_returns_more_than_zero_but_more_than_twenty():
    n = roll()
    assert n > 0 and n <= 20
