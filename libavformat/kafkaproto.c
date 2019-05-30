/*
 * Kafka network protocol
 * Copyright (c) 2019 Shaofei Wang
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Kafka protocol based producer
 */

#include "avformat.h"
#include "url.h"

#include "librdkafka/rdkafka.h"

typedef struct KafkaContext {
    rd_kafka_t *rk;         /* Producer instance handle */
    rd_kafka_conf_t *conf;  /* Configuration object */
    rd_kafka_topic_t *rkt;  /* Topic object */
} KafkaContext;

static void dr_msg_cb (rd_kafka_t *rk,
                       const rd_kafka_message_t *rkmessage, void *opaque) {
    if (rkmessage->err)
        fprintf(stderr, "%% Message delivery failed: %s\n",
                rd_kafka_err2str(rkmessage->err));
    /* Avoid too much print mesg
    else
        fprintf(stderr, "%% Message delivered (%zd bytes, "
                "partition %"PRId32")\n",
                rkmessage->len, rkmessage->partition);
                */
}

static int kafka_open(URLContext *s, const char *uri, int flags, AVDictionary **opts)
{
    KafkaContext *kc = s->priv_data;

    char proto[8], hostname[256], path[1024], auth[100], brokers[256], errstr[512], *topic;
    int port;

    av_url_split(proto, sizeof(proto), auth, sizeof(auth),
                 hostname, sizeof(hostname), &port,
                 path, sizeof(path), s->filename);
    topic = strrchr(s->filename, '/') + 1;

    kc->conf = rd_kafka_conf_new();
    port ? snprintf(brokers, 256, "%s:%d", hostname, port)
        : snprintf(brokers, 256, "%s:9092", hostname);

    /* Set bootstrap broker(s) as a comma-separated list of
     * host or host:port (default port 9092).
     * librdkafka will use the bootstrap brokers to acquire the full
     * set of brokers from the cluster. */
    if (rd_kafka_conf_set(kc->conf, "bootstrap.servers", brokers,
                          errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
        av_log(s, AV_LOG_ERROR, "%s\n", errstr);
        return AVERROR_UNKNOWN;
    }

    rd_kafka_conf_set_dr_msg_cb(kc->conf, dr_msg_cb);

    kc->rk = rd_kafka_new(RD_KAFKA_PRODUCER, kc->conf, errstr, sizeof(errstr));
    if (!kc->rk) {
        av_log(s, AV_LOG_ERROR,
                "%% Failed to create new producer: %s\n", errstr);
        return AVERROR_UNKNOWN;
    }

    kc->rkt = rd_kafka_topic_new(kc->rk, topic, NULL);
    if (!kc->rkt) {
        av_log(s, AV_LOG_ERROR,
                "%% Failed to create topic object: %s\n",
                rd_kafka_err2str(rd_kafka_last_error()));
        rd_kafka_destroy(kc->rk);
        return AVERROR_UNKNOWN;
    }

    return 0;
}

static int kafka_close(URLContext *h)
{
    KafkaContext *kc = h->priv_data;

    rd_kafka_flush(kc->rk, 10*1000 /* wait for max 10 seconds */);

    rd_kafka_topic_destroy(kc->rkt);

    rd_kafka_destroy(kc->rk);

    return 0;
}

static int kafka_write(URLContext *s, const uint8_t *buf, int size)
{
    KafkaContext *kc = s->priv_data;
    rd_kafka_t *rk = kc->rk;
    rd_kafka_topic_t *rkt = kc->rkt;

    if (size == 0) {
        /* Empty line: only serve delivery reports */
        rd_kafka_poll(rk, 0/*non-blocking */);
        return 0;
    }

retry:
    if (rd_kafka_produce(
        /* Topic object */
        rkt,
        /* Use builtin partitioner to select partition*/
        RD_KAFKA_PARTITION_UA,
        /* Make a copy of the payload. */
        RD_KAFKA_MSG_F_COPY,
        /* Message payload (value) and length */
        buf, size,
        /* Optional key and its length */
        NULL, 0,
        /* Message opaque, provided in
         * delivery report callback as
         * msg_opaque. */
        NULL) == -1) {

        /**
         * Failed to *enqueue* message for producing.
         */
        av_log(s, AV_LOG_ERROR,
                "%% Failed to produce to topic %s: %s\n",
                rd_kafka_topic_name(rkt),
                rd_kafka_err2str(rd_kafka_last_error()));

        /* Poll to handle delivery reports */
        if (rd_kafka_last_error() ==
            RD_KAFKA_RESP_ERR__QUEUE_FULL) {
                /* If the internal queue is full, wait for
                 * messages to be delivered and then retry.
                 * The internal queue represents both
                 * messages to be sent and messages that have
                 * been sent or failed, awaiting their
                 * delivery report callback to be called.
                 *
                 * The internal queue is limited by the
                 * configuration property
                 * queue.buffering.max.messages */
                rd_kafka_poll(rk, 1000/*block for max 1000ms*/);
                goto retry;
        }
    } else {
        rd_kafka_poll(rk, 0/*non-blocking*/);
    }
    return size;
}

#define KAFKA_PROTOCOL(flavor)                    \
static const AVClass flavor##_class = {           \
    .class_name = #flavor,                        \
    .item_name  = av_default_item_name,           \
    .version    = LIBAVUTIL_VERSION_INT,          \
};                                                \
                                                  \
const URLProtocol ff_##flavor##_protocol = {      \
    .name           = "kafka",                    \
    .url_open       = kafka_open,                 \
    .url_write      = kafka_write,                \
    .url_close      = kafka_close,                \
    .priv_data_size = sizeof(KafkaContext),     \
    .flags          = URL_PROTOCOL_FLAG_NETWORK,  \
};

KAFKA_PROTOCOL(rdkafka)
