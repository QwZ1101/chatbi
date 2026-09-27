-- ---------------------------------------------------------------
-- 股票历史日线行情表（对应 stock_history.xlsx 的「汇总」Sheet）
-- 字符集 utf8mb4，引擎 InnoDB
-- 字段与 fetch_stock_history.py 中的英文列一一对应
-- ---------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS `chatbi`
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE `chatbi`;

DROP TABLE IF EXISTS `stock_daily`;

CREATE TABLE `stock_daily` (
    `id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    `stock_name`  VARCHAR(32)     NOT NULL                COMMENT '股票名称，如 贵州茅台',
    `ts_code`     VARCHAR(16)     NOT NULL                COMMENT '股票代码，如 600519.SH',
    `trade_date`  DATE            NOT NULL                COMMENT '交易日期',
    `open`        DECIMAL(18, 4)  DEFAULT NULL            COMMENT '开盘价',
    `high`        DECIMAL(18, 4)  DEFAULT NULL            COMMENT '最高价',
    `low`         DECIMAL(18, 4)  DEFAULT NULL            COMMENT '最低价',
    `close`       DECIMAL(18, 4)  DEFAULT NULL            COMMENT '收盘价',
    `pre_close`   DECIMAL(18, 4)  DEFAULT NULL            COMMENT '前收盘价',
    `change`      DECIMAL(18, 4)  DEFAULT NULL            COMMENT '涨跌额',
    `pct_chg`     DECIMAL(12, 4)  DEFAULT NULL            COMMENT '涨跌幅(%)',
    `vol`         DECIMAL(20, 2)  DEFAULT NULL            COMMENT '成交量(手)',
    `amount`      DECIMAL(22, 4)  DEFAULT NULL            COMMENT '成交额(千元)',
    `created_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP                    COMMENT '记录创建时间',
    `updated_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '记录更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_code_date` (`ts_code`, `trade_date`),
    KEY `idx_trade_date` (`trade_date`),
    KEY `idx_stock_name` (`stock_name`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '股票历史日线行情表';
