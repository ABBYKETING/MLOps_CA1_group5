-- SP-buy fraud detection - RDS MySQL table schemas (PBI 4, Task #42)
--
-- Run with:  python src/database.py
-- Safe to run again: every table uses CREATE TABLE IF NOT EXISTS.
--
-- Design rules (from ST1519_CA1_Data_Cleaning_Assumptions and src/data_processing.py):
--   * A customer is identified by (country_code, customer_id).
--   * An order is identified by (country_code, order_id)   - Assumption 11.
--   * The orders file has no customer_id; the label links an order to its customer.
--   * refund_value is stored for analytics only and must never be a model feature (Assumptions 15-16).
--   * Rows that cannot safely be cleaned go to rejected_records with a reason (Assumption 18).
--   * "Latest wins" across batches (PO answers Q4 and Q5): a customer, order or label that is
--     already in the database is UPDATED with the newer batch, so there is one current record
--     each and a corrected fraud label is not lost.
--   * Load order: customers -> orders -> fraud_labels (labels reference both).


-- Customer snapshot. Re-sent customers are UPDATED (Task #45), so we keep
-- which batch first created the row and which batch last changed it.
CREATE TABLE IF NOT EXISTS customers (
    country_code                      CHAR(2)       NOT NULL,
    customer_id                       VARCHAR(32)   NOT NULL,
    mobile_verified                   BOOLEAN       NOT NULL,
    num_orders_last_50days            INT UNSIGNED  NOT NULL,
    num_cancelled_orders_last_50days  INT UNSIGNED  NOT NULL,
    num_refund_orders_last_50days     INT UNSIGNED  NOT NULL,
    total_payment_last_50days         DOUBLE        NOT NULL,
    num_associated_customers          INT UNSIGNED  NOT NULL,
    first_order_datetime              DATETIME      NULL,          -- Assumption 8: may be missing
    first_loaded_batch                VARCHAR(64)   NOT NULL,
    last_updated_batch                VARCHAR(64)   NOT NULL,
    created_at                        TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                        TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (country_code, customer_id),
    CONSTRAINT chk_customers_payment CHECK (total_payment_last_50days >= 0)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


-- Orders. An order that arrives again in a later batch replaces the older
-- record ("latest wins", Task #46), so there is always one current record per order.
CREATE TABLE IF NOT EXISTS orders (
    country_code       CHAR(2)       NOT NULL,
    order_id           VARCHAR(32)   NOT NULL,
    collect_type       VARCHAR(10)   NOT NULL,
    payment_method     VARCHAR(40)   NOT NULL,
    order_value        DOUBLE        NOT NULL,
    num_items_ordered  INT UNSIGNED  NOT NULL,
    refund_value       DOUBLE        NOT NULL,                     -- analytics only, never a model feature
    order_date         DATE          NOT NULL,
    first_loaded_batch VARCHAR(64)   NOT NULL,
    last_updated_batch VARCHAR(64)   NOT NULL,
    created_at         TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (country_code, order_id),
    INDEX idx_orders_order_date (order_date),                     -- time-based analytics and the chronological split
    CONSTRAINT chk_orders_collect_type CHECK (collect_type IN ('Delivery', 'Pickup')),
    CONSTRAINT chk_orders_value  CHECK (order_value >= 0),
    CONSTRAINT chk_orders_items  CHECK (num_items_ordered >= 1),
    CONSTRAINT chk_orders_refund CHECK (refund_value >= 0)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


-- Fraud labels: one label per order, linking the order to its customer.
-- A label sent again in a later batch replaces the older one (e.g. a corrected is_fraud).
-- The foreign keys stop a label from being stored without its order and customer
-- (Assumption 14), so the database always holds complete training rows.
CREATE TABLE IF NOT EXISTS fraud_labels (
    country_code  CHAR(2)      NOT NULL,
    order_id      VARCHAR(32)  NOT NULL,
    customer_id   VARCHAR(32)  NOT NULL,
    is_fraud            TINYINT      NOT NULL,
    first_loaded_batch  VARCHAR(64)  NOT NULL,
    last_updated_batch  VARCHAR(64)  NOT NULL,
    created_at          TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (country_code, order_id),
    CONSTRAINT chk_labels_is_fraud CHECK (is_fraud IN (0, 1)),
    CONSTRAINT fk_labels_order    FOREIGN KEY (country_code, order_id)
        REFERENCES orders (country_code, order_id),
    CONSTRAINT fk_labels_customer FOREIGN KEY (country_code, customer_id)
        REFERENCES customers (country_code, customer_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


-- Rows that could not be cleaned safely, kept with the reason (Assumption 18).
-- raw_record holds the original row exactly as received, so nothing is lost.
CREATE TABLE IF NOT EXISTS rejected_records (
    rejected_id   BIGINT        NOT NULL AUTO_INCREMENT,
    batch_name    VARCHAR(64)   NOT NULL,
    source_table  VARCHAR(20)   NOT NULL,
    country_code  VARCHAR(16)   NULL,                              -- raw value, may be invalid
    record_id     VARCHAR(64)   NULL,                              -- customer_id or order_id, if present
    reason        VARCHAR(255)  NOT NULL,
    raw_record    JSON          NOT NULL,
    rejected_at   TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (rejected_id),
    INDEX idx_rejected_batch (batch_name),
    CONSTRAINT chk_rejected_source CHECK (source_table IN ('customers', 'orders', 'fraud_labels'))
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


-- One row per ETL run of a batch ("base" or the postfix of new files, e.g. "XXX").
-- Shows what each run did (how many rows were inserted, updated or rejected)
-- and lets analytics/retraining check how fresh the data is.
CREATE TABLE IF NOT EXISTS etl_batches (
    batch_name                 VARCHAR(64)   NOT NULL,
    status                     VARCHAR(20)   NOT NULL,
    started_at                 DATETIME      NOT NULL,
    finished_at                DATETIME      NULL,
    customers_inserted         INT UNSIGNED  NOT NULL DEFAULT 0,
    customers_updated          INT UNSIGNED  NOT NULL DEFAULT 0,
    orders_inserted            INT UNSIGNED  NOT NULL DEFAULT 0,
    orders_updated             INT UNSIGNED  NOT NULL DEFAULT 0,
    labels_inserted            INT UNSIGNED  NOT NULL DEFAULT 0,
    labels_updated             INT UNSIGNED  NOT NULL DEFAULT 0,
    rows_rejected              INT UNSIGNED  NOT NULL DEFAULT 0,
    error_message              TEXT          NULL,
    PRIMARY KEY (batch_name),
    CONSTRAINT chk_batches_status CHECK (status IN ('running', 'completed', 'failed'))
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;
