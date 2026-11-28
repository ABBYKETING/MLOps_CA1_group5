# Payment and order-pattern analytics - Chu

Data: 949,969 labelled orders; 104,433 fraud labels (10.99%).
Order dates: 2022-11-12 to 2023-05-11. Unlabelled orders excluded: 24.

## Q1. Which payment methods merit closer monitoring?

Among methods with at least 100 labelled orders, **jazzcash wallet** has the highest observed fraud rate: 48.90% (1,227/2,509; 95% interval 46.95-50.86%).
**payment on delivery** contributes the most fraud-labelled orders: 50,274 (48.14% of observed fraud), with a 16.08% rate.
Decision: use both fraud concentration and transaction volumes to prioritise review; check the country breakdown before changing payment-method controls.

## Q2. Which payment/value combinations merit review?

The largest within-method lift among supported groups is **cybersource, (50, 100]**: 59.75% fraud (190/318), 18.19 times that method's 3.29% baseline.
Decision: investigate the supported payment/value combinations with elevated rates and meaningful fraud counts; a larger order is not automatically riskier. Consult country-specific results and intervals.

## Q3. Does collection type change the observed pattern?

The largest absolute gap with both groups supported is **antfinancial bkash**: Delivery 22.48% (n=16,140) versus Pickup 58.26% (n=1,749). Pickup has the higher point estimate; signed Delivery-minus-Pickup gap -35.78 percentage points.
Decision: review payment/collection combinations rather than treating all delivery or pickup orders as equally risky. A missing comparison group means insufficient evidence, not zero risk.

## Definitions and interpretation limits

- Fraud rate = fraud-labelled orders / labelled orders in the same group; counts are order-weighted.
- Intervals are 95% Wilson intervals for descriptive binomial uncertainty; repeated customers and non-random labels mean they are not a guarantee about future populations.
- The 100-order rule controls highlighted examples only. All groups, including rare methods, remain in tables and charts.
- Value ranges use SGD as specified in the brief. Zero-value orders are valid and retained separately.
- `unknown` is an unrecorded-payment segment; `no payment` is retained as a valid recorded category.
- Payment methods have different country mixes. Country tables support stratified checking; pooled gaps are not causal effects.
- These are retrospective labelled-order associations, not fraud predictions, proof of causation or instructions to block transactions.