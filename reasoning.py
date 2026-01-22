def generate_reasoning(features):
    """
    Generates human-friendly fraud reasons
    from behavioral feature values.
    """

    reasons = []

    if features["amt_deviation"] > 5000:
        reasons.append(
            "The transaction amount is very different from what the customer usually spends."
        )

    if features["txn_count_cust"] > 10:
        reasons.append(
            "There have been many transactions from this customer in a short period of time."
        )

    if features["cust_category_count"] == 0:
        reasons.append(
            "The customer has not made purchases in this type of store before."
        )

    if features["distance_from_home"] > 100:
        reasons.append(
            "The transaction is happening far from the customer’s usual location."
        )

    if not reasons:
        reasons.append(
            "The transaction shows a combination of unusual patterns compared to the customer’s normal behavior."
        )

    # Limit explanations (important for APIs)
    return reasons[:3]
