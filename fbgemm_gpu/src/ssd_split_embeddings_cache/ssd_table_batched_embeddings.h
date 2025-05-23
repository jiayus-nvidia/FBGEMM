/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include <memory>

#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Task.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <torch/nn/init.h>
#ifdef FBGEMM_FBCODE
#include "common/strings/UUID.h"
#include "common/time/Time.h"
#include "fb_rocksdb/DBMonitor/DBMonitor.h"
#include "fb_rocksdb/FbRocksDb.h"
#include "rocks/utils/FB303Stats.h"
#endif
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "initializer.h"
#include "kv_db_table_batched_embeddings.h"
#include "kv_tensor_wrapper.h"
#include "torch/csrc/autograd/record_function_ops.h"

namespace ssd {

using namespace at;

#ifdef FBGEMM_FBCODE
constexpr size_t num_ssd_drives = 8;
const std::string ssd_mount_point = "/data00_nvidia";
const size_t base_port = 136000;
#endif

// mem usage propertiese
// -- block cache usage
// -- Indexes and filter blocks usage
// -- Memtable usage
// -- Blocks pinned by iterators usage
// for details, checkout https://fburl.com/by9kyk12
const std::vector<std::string> rocks_db_mem_properties = {
    "rocksdb.block-cache-usage",
    "rocksdb.estimate-table-readers-mem",
    "rocksdb.cur-size-all-mem-tables",
    "rocksdb.block-cache-pinned-usage",
};

class EmbeddingRocksDB;
using snapshot_ptr_t = const rocksdb::Snapshot*;
// @lint-ignore CLANGTIDY cppcoreguidelines-special-member-functions
class SnapshotHandle {
 public:
  explicit SnapshotHandle(EmbeddingRocksDB* db);
  ~SnapshotHandle();
  void release();
  snapshot_ptr_t get_snapshot_for_shard(size_t shard) const;

 private:
  friend class EmbeddingRocksDB;

  EmbeddingRocksDB* db_;
  std::vector<snapshot_ptr_t> shard_snapshots_;
}; // class SnapshotHandle

/// @ingroup embedding-ssd
///
/// @brief An implementation of EmbeddingKVDB for RocksDB
///
class EmbeddingRocksDB : public kv_db::EmbeddingKVDB {
 public:
  explicit EmbeddingRocksDB(
      std::string path,
      int64_t num_shards,
      int64_t num_threads,
      int64_t memtable_flush_period,
      int64_t memtable_flush_offset,
      int64_t l0_files_per_compact,
      int64_t max_D,
      int64_t rate_limit_mbps,
      int64_t size_ratio,
      int64_t compaction_trigger,
      int64_t write_buffer_size,
      int64_t max_write_buffer_num,
      float uniform_init_lower,
      float uniform_init_upper,
      int64_t row_storage_bitwidth = 32,
      int64_t cache_size = 0,
      bool use_passed_in_path = false,
      int64_t tbe_unqiue_id = 0,
      int64_t l2_cache_size_gb = 0,
      bool enable_async_update = false,
      bool enable_raw_embedding_streaming = false,
      int64_t res_store_shards = 0,
      int64_t res_server_port = 0,
      std::vector<std::string> table_names = {},
      std::vector<int64_t> table_offsets = {},
      const std::vector<int64_t>& table_sizes = {},
      std::optional<at::Tensor> table_dims = std::nullopt,
      std::optional<at::Tensor> hash_size_cumsum = std::nullopt)
      : kv_db::EmbeddingKVDB(
            num_shards,
            max_D,
            l2_cache_size_gb,
            tbe_unqiue_id,
            row_storage_bitwidth / 8,
            enable_async_update,
            enable_raw_embedding_streaming,
            res_store_shards,
            res_server_port,
            std::move(table_names),
            std::move(table_offsets),
            table_sizes),
        auto_compaction_enabled_(true),
        max_D_(max_D),
        elem_size_(row_storage_bitwidth / 8) {
    class Int64Comparator : public rocksdb::Comparator {
     public:
      const char* Name() const override {
        return "Int64Comparator";
      }

      int Compare(const rocksdb::Slice& a, const rocksdb::Slice& b)
          const override {
        int64_t key_a = *reinterpret_cast<const int64_t*>(a.data());
        int64_t key_b = *reinterpret_cast<const int64_t*>(b.data());
        if (key_a < key_b) {
          return -1;
        }
        if (key_a > key_b) {
          return 1;
        }
        return 0;
      }

      void FindShortestSeparator(std::string*, const rocksdb::Slice&)
          const override {}
      void FindShortSuccessor(std::string*) const override {}
    };

    // TODO: lots of tunables. NNI or something for this?
    rocksdb::Options options;
    options.comparator = new Int64Comparator();
    options.create_if_missing = true;

    // TODO: probably not very compressible.
    options.compression = rocksdb::kNoCompression;

    // Lots of free memory on the TC, use large write buffers.
    // max_write_buffer_num is per rocksdb shard level, write_buffer_size is tbe
    // level to calc individual buffer size we need to have total buffer size
    // per tbe / # db shards / # buffer per shards
    int64_t write_buffer_size_per_buffer =
        int64_t(write_buffer_size / num_shards / max_write_buffer_num);
    options.write_buffer_size = write_buffer_size_per_buffer;
    options.max_write_buffer_number = max_write_buffer_num;
    options.min_write_buffer_number_to_merge = 2;
    options.target_file_size_base = int64_t(2) * 1024 * 1024 * 1024;

    options.compaction_style = rocksdb::kCompactionStyleUniversal;
    options.compaction_options_universal.size_ratio = size_ratio;
    options.compaction_options_universal.min_merge_width = 2;
    // size amplification ratio = (size(R1) + size(R2) + ... size(Rn-1)) /
    // size(Rn)
    options.compaction_options_universal.max_size_amplification_percent = 400;
    options.level0_file_num_compaction_trigger = compaction_trigger;
    options.level0_slowdown_writes_trigger = 32;
    options.level0_stop_writes_trigger = 64;
    options.prefix_extractor.reset(
        rocksdb::NewFixedPrefixTransform(sizeof(int64_t)));
    // Partial Pipeline Options
    // options.allow_concurrent_memtable_write = false;
    // options.inplace_update_support = true;
    // Full Pipeline Options
    options.allow_concurrent_memtable_write = false;
    options.enable_write_thread_adaptive_yield = true;
    // inplace_update_support = false means we will apend kv pair in write
    // buffer even we saw duplications, this quickly fills up the buffer and
    // causing flush set this to true to make update on the existing key
    // allow_concurrent_memtable_write is toggled in pair with
    // inplace_update_support
    options.inplace_update_support = false;
    options.avoid_unnecessary_blocking_io = true;

    options.use_direct_reads = true;
    options.use_direct_io_for_flush_and_compaction = true;

    if (rate_limit_mbps > 0) {
      rate_limiter_.reset(
          rocksdb::NewGenericRateLimiter(rate_limit_mbps * 1024 * 1024));
    }
    options.rate_limiter = rate_limiter_;

    // TODO: use fb303?
#ifdef FBGEMM_FBCODE
    options.statistics =
        std::make_shared<facebook::rocks::FB303Stats>("tbe_metrics");
#else
    options.statistics = rocksdb::CreateDBStatistics();
#endif
    options.stats_dump_period_sec = 600;

    // no bloom filter on the last level, checkout https://fburl.com/ne99girf
    options.optimize_filters_for_hits = true;

    rocksdb::BlockBasedTableOptions table_options;

    if (cache_size > 0) {
      table_options.block_cache = rocksdb::NewLRUCache(cache_size);
      table_options.cache_index_and_filter_blocks = true;
    } else {
      table_options.no_block_cache = true;
    }

    table_options.index_type = rocksdb::BlockBasedTableOptions::kHashSearch;
    table_options.data_block_index_type =
        rocksdb::BlockBasedTableOptions::kDataBlockBinaryAndHash;
    table_options.data_block_hash_table_util_ratio = 0.75;
    table_options.checksum = rocksdb::ChecksumType::kNoChecksum;
    table_options.format_version = 5;
    table_options.read_amp_bytes_per_bit = 1;

    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(16));
    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));
    options.memtable_prefix_bloom_size_ratio = 0.05;
    options.memtable_whole_key_filtering = true;
    options.max_background_jobs = num_threads;
    // maximum number of concurrent flush operations
    options.max_background_flushes = num_threads;
    options.env->SetBackgroundThreads(4, rocksdb::Env::HIGH);
    options.env->SetBackgroundThreads(1, rocksdb::Env::LOW);
    options.max_open_files = -1;

    initialize_dbs(num_shards, path, options, use_passed_in_path);
    initialize_initializers(
        num_shards,
        max_D,
        uniform_init_lower,
        uniform_init_upper,
        row_storage_bitwidth);
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(num_shards);
    ro_.verify_checksums = false;
    ro_.async_io = true;
    wo_.disableWAL = true;
    wo_.sync = false;

    // Setup staggered manual compaction data members
    memtable_flush_period_ = memtable_flush_period;
    if (memtable_flush_period_ > 0) {
      done_staggered_flushes_ = false;
      memtable_flush_offset_ = memtable_flush_offset;
      l0_files_per_compact_ = l0_files_per_compact;
      compaction_period_ = memtable_flush_period_ * l0_files_per_compact *
          options.min_write_buffer_number_to_merge;
      int64_t period_per_shard = memtable_flush_period_ / num_shards;
      CHECK_GT(period_per_shard, 0);
      // We want to stagger memory flushes (and then later
      // stagger all compactions)

      for (int64_t i = 0; i < num_shards; i++) {
        shard_flush_compaction_deadlines_.push_back(
            memtable_flush_offset_ + (i * period_per_shard));
      }
    }
    if (table_dims.has_value()) {
      TORCH_CHECK(table_dims->dim() == 1);
      TORCH_CHECK(table_dims->dtype() == at::ScalarType::Long);
      TORCH_CHECK(table_dims->is_contiguous());
      TORCH_CHECK(table_dims->device().is_cpu());
      TORCH_CHECK(hash_size_cumsum.has_value());
      TORCH_CHECK(hash_size_cumsum->dim() == 1);
      TORCH_CHECK(hash_size_cumsum->dtype() == at::ScalarType::Long);
      TORCH_CHECK(hash_size_cumsum->is_contiguous());
      TORCH_CHECK(hash_size_cumsum->device().is_cpu());
      TORCH_CHECK(
          table_dims->numel() + 1 == hash_size_cumsum->numel(),
          "hash_size_cumsum length must be one more than table_dims length, but got ",
          hash_size_cumsum->numel(),
          " and ",
          table_dims->numel());
      sub_table_dims_.assign(
          table_dims->data_ptr<int64_t>(),
          table_dims->data_ptr<int64_t>() + table_dims->numel());
      sub_table_hash_cumsum_.assign(
          hash_size_cumsum->data_ptr<int64_t>() + 1, // skip the first 0
          hash_size_cumsum->data_ptr<int64_t>() + hash_size_cumsum->numel());
    }
  }

  ~EmbeddingRocksDB() override {
    // clear all the snapshots if not released
    if (snapshots_.size() > 0) {
      LOG(WARNING)
          << snapshots_.size()
          << " snapshots have not been released when db is closing. Releasing them now.";
    }
    snapshots_.clear();
    for (auto shard = 0; shard < dbs_.size(); ++shard) {
      dbs_[shard]->Close();
    }
  }

  void initialize_dbs(
      int64_t num_shards,
      std::string path,
      rocksdb::Options& options,
      bool use_passed_in_path) {
#ifdef FBGEMM_FBCODE
    std::string used_path = "";
    auto serviceInfo = std::make_shared<facebook::fb_rocksdb::ServiceInfo>();
    serviceInfo->oncall = "pyper_training";
    serviceInfo->service_name = "ssd_offloading_rocksb";
    auto db_monitor_options = facebook::fb_rocksdb::DBMonitorOptions();
    db_monitor_options.fb303Prefix = "tbe_metrics";

    std::string tbe_uuid = "";
    if (!use_passed_in_path) {
      path = ssd_mount_point;
      tbe_uuid = facebook::strings::generateUUID();
    }
#endif
    for (auto i = 0; i < num_shards; ++i) {
#ifdef FBGEMM_FBCODE
      int ssd_drive_idx = i % num_ssd_drives;
      std::string ssd_idx_tbe_id_str = "";
      if (!use_passed_in_path) {
        ssd_idx_tbe_id_str =
            std::to_string(ssd_drive_idx) + std::string("/") + tbe_uuid;
      }
      auto shard_path =
          path + ssd_idx_tbe_id_str + std::string("_shard") + std::to_string(i);
      used_path += shard_path + ", ";
#else
      auto shard_path = path + std::string("/shard_") + std::to_string(i);
#endif
      rocksdb::DB* db;

#ifdef FBGEMM_FBCODE
      auto s = facebook::fb_rocksdb::openRocksDB(
          options,
          shard_path,
          &db,
          serviceInfo,
          facebook::fb_rocksdb::getDefaultProfileOptions(),
          db_monitor_options);
#else
      auto s = rocksdb::DB::Open(options, shard_path, &db);
#endif
      if (!s.ok() && s.code() == rocksdb::Status::kInvalidArgument &&
          (options.use_direct_reads ||
           options.use_direct_io_for_flush_and_compaction)) {
        LOG(WARNING)
            << "Warning, Requested DirectIO, but not supported on destination: "
            << shard_path;
        options.use_direct_reads = false;
        options.use_direct_io_for_flush_and_compaction = false;
        LOG(WARNING)
            << "Trying again, any subsequent failures will be fatal...";
        s = rocksdb::DB::Open(options, shard_path, &db);
      }
      CHECK(s.ok()) << s.ToString();
      dbs_.emplace_back(db);
    }
#ifdef FBGEMM_FBCODE
    LOG(INFO) << "TBE actual used_path: " << used_path;
#endif
  }

  void initialize_initializers(
      int64_t num_shards,
      int64_t max_D,
      float uniform_init_lower,
      float uniform_init_upper,
      int64_t row_storage_bitwidth) {
    for (auto i = 0; i < num_shards; ++i) {
      auto* gen = at::check_generator<at::CPUGeneratorImpl>(
          at::detail::getDefaultCPUGenerator());
      {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        initializers_.push_back(std::make_unique<Initializer>(
            gen->random64(),
            max_D,
            uniform_init_lower,
            uniform_init_upper,
            row_storage_bitwidth));
      }
    }
  }

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    return get_kv_db_async_impl</*use_iterator=*/false>(
        indices,
        weights,
        count,
        /*snapshot_handle=*/nullptr);
  }

  folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const kv_db::RocksdbWriteMode w_mode =
          kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ) override {
    RECORD_USER_SCOPE("EmbeddingRocksDB::set");
#ifdef FBGEMM_FBCODE
    auto start_ts = facebook::WallClockUtil::NowInUsecFast();
#endif
    std::vector<folly::Future<folly::Unit>> futures;
    auto count_ = count.scalar_type() == at::ScalarType::Long
        ? *(count.data_ptr<int64_t>())
        : *(count.data_ptr<int32_t>());

    for (auto shard = 0; shard < dbs_.size(); ++shard) {
      auto f =
          folly::via(executor_.get())
              .thenValue([=, &indices, &weights](folly::Unit) {
                FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
                    weights.scalar_type(), "ssd_set", [&] {
                      using value_t = scalar_t;
                      FBGEMM_DISPATCH_INTEGRAL_TYPES(
                          indices.scalar_type(), "ssd_set", [&] {
                            using index_t = scalar_t;
                            CHECK(indices.is_contiguous());
                            CHECK(weights.is_contiguous());
                            auto indices_acc = indices.accessor<index_t, 1>();
                            auto D = weights.size(1);
                            CHECK_EQ(indices.size(0), weights.size(0));
                            {
                              rocksdb::WriteBatch batch(
                                  (2 * (count_ + dbs_.size() - 1) /
                                   dbs_.size()) *
                                  (sizeof(index_t) + sizeof(value_t) * D));
                              for (auto i = 0; i < count_; ++i) {
                                if (indices_acc[i] < 0) {
                                  continue;
                                }
                                if (kv_db_utils::hash_shard(
                                        indices_acc[i], dbs_.size()) != shard) {
                                  continue;
                                }
                                batch.Put(
                                    rocksdb::Slice(
                                        reinterpret_cast<const char*>(
                                            &(indices.data_ptr<index_t>()[i])),
                                        sizeof(index_t)),
                                    rocksdb::Slice(
                                        reinterpret_cast<const char*>(
                                            &(weights
                                                  .data_ptr<value_t>()[i * D])),
                                        D * sizeof(value_t)));
                              }
                              auto s = dbs_[shard]->Write(wo_, &batch);
                              CHECK(s.ok());
                            }
                          });
                    });
              });
      futures.push_back(std::move(f));
    }
    // co_await folly::coro::collectAllRange(std::move(tasks));
#ifdef FBGEMM_FBCODE
    auto duration = facebook::WallClockUtil::NowInUsecFast() - start_ts;
    switch (w_mode) {
      case kv_db::RocksdbWriteMode::BWD_L1_CNFLCT_MISS_WRITE_BACK:
        bwd_l1_cnflct_miss_write_back_dur_ += duration;
        break;
      case kv_db::RocksdbWriteMode::FWD_L1_EVICTION:
        fwd_l1_eviction_dur_ += duration;
        break;
      case kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ:
        fwd_rocksdb_read_dur_ += duration;
        break;
      case kv_db::RocksdbWriteMode::FLUSH:
        flush_write_dur_ += duration;
        break;
    }
#endif
    return folly::collect(futures);
  }

  bool is_valid_snapshot(const SnapshotHandle* snapshot_handle) const {
    return snapshots_.find(snapshot_handle) != snapshots_.end();
  }

  int64_t get_snapshot_count() const {
    return snapshots_.size();
  }

  const SnapshotHandle* create_snapshot() {
    const auto num_snapshots = snapshots_.size();
    if (num_snapshots > 0) {
      std::cerr << "WARNING: create_snapshot found " << num_snapshots
                << " other snapshots" << std::endl;
    }

    auto handle = std::make_unique<SnapshotHandle>(this);
    auto handlePtr = handle.get();
    snapshots_[handlePtr] = std::move(handle);
    return handlePtr;
  }

  void release_snapshot(const SnapshotHandle* snapshot_handle) {
    CHECK(is_valid_snapshot(snapshot_handle));
    LOG(INFO) << "Snapshot " << snapshot_handle << " released";
    snapshots_.erase(snapshot_handle);
  }

  /// get existing ids from rocksdb in a given range at a given rocksdb
  /// snapshot
  /// @param start_id the inclusive start stored_id of the range
  /// @param end_id the exclusive end stored_id of the range
  /// @param id_offset to translate the rocksdb stored id(linearized id) to
  /// original id(unlinearized id)
  /// @return unlinearized ids in random order
  at::Tensor get_keys_in_range_by_snapshot(
      int64_t start_id,
      int64_t end_id,
      int64_t id_offset,
      const SnapshotHandle* snapshot_handle) {
    std::vector<folly::Future<folly::Unit>> futures;

    const auto start_key = rocksdb::Slice(
        reinterpret_cast<const char*>(&start_id), sizeof(int64_t));
    // [db shard -> [original_ids]]
    std::vector<std::vector<int64_t>> keys_in_db_shards(dbs_.size());
    for (auto& keys : keys_in_db_shards) {
      keys.reserve(1 * 1024 * 1024); // reserve 1M items space
    }

    // parallel loop through all db shards by iterating from start id to end id
    for (auto shard = 0; shard < dbs_.size(); ++shard) {
      // Get a snapshot for the shard
      snapshot_ptr_t snapshot = snapshot_handle == nullptr
          ? nullptr
          : snapshot_handle->get_snapshot_for_shard(shard);
      auto local_ro = ro_;
      local_ro.snapshot = snapshot;
      auto f =
          folly::via(executor_.get())
              .thenValue([&, local_ro, shard](folly::Unit) {
                auto* dcf = dbs_[shard]->DefaultColumnFamily();

                auto iterator_ro = local_ro;
                iterator_ro.total_order_seek = true; // disable prefix filter
                auto it = dbs_[shard]->NewIterator(iterator_ro, dcf);
                it->Seek(start_key);
                while (true) {
                  CHECK(it->status().ok());
                  if (!it->Valid()) {
                    CHECK(it->status().ok());
                    // the iterator reaches the end of the data
                    break;
                  }
                  int64_t stored_value =
                      *reinterpret_cast<const int64_t*>(it->key().data());
                  if (stored_value >= end_id) {
                    // go beyond the end of the range
                    break;
                  }
                  keys_in_db_shards[shard].push_back(stored_value - id_offset);
                  it->Next();
                }

                delete it;
              });
      futures.push_back(std::move(f));
    }
    folly::collect(futures).wait();
    int64_t total_num = 0;

    // calc total num for tensor allocation and count_in_ranges
    for (const auto& keys : keys_in_db_shards) {
      total_num += keys.size();
    }

    at::Tensor returned_keys = at::empty(
        {total_num, 1}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    auto key_ptr = returned_keys.data_ptr<int64_t>();
    int64_t offset = 0;
    for (const auto& keys : keys_in_db_shards) {
      std::copy(keys.begin(), keys.end(), &key_ptr[offset]);
      offset += keys.size();
    }
    return returned_keys;
  }

  void get_range_from_snapshot(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length,
      const SnapshotHandle* snapshot_handle) override {
    const auto seq_indices =
        at::arange(start, start + length, at::TensorOptions().dtype(at::kLong));
    const auto count = at::tensor({length}, at::ScalarType::Long);

    get_kv_db_async_impl</*use_iterator=*/true>(
        seq_indices, weights, count, snapshot_handle)
        .wait();
  }

  void set_kv_to_storage(const at::Tensor& ids, const at::Tensor& weights) {
    const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
    folly::coro::blockingWait(set_kv_db_async(ids, weights, count));
  }

  void get_kv_from_storage_by_snapshot(
      const at::Tensor& ids,
      const at::Tensor& weights,
      const SnapshotHandle* snapshot_handle) override {
    const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
    get_kv_db_async_impl</*use_iterator=*/false>(
        ids, weights, count, snapshot_handle)
        .wait();
  }

  virtual rocksdb::Status set_rocksdb_option(
      int shard,
      const std::string& key,
      const std::string& value) {
    return dbs_[shard]->SetOptions({{key, value}});
  }

  bool is_auto_compaction_enabled() {
    return auto_compaction_enabled_;
  }

  void toggle_compaction(bool enable) {
    int max_retries = 10;
    std::vector<folly::Future<bool>> futures;
    for (auto shard = 0; shard < dbs_.size(); ++shard) {
      auto f = folly::via(executor_.get()).thenValue([=](folly::Unit) -> bool {
        for (int attempt = 0; attempt < max_retries; ++attempt) {
          auto val = enable ? "false" : "true";
          auto s = set_rocksdb_option(shard, "disable_auto_compactions", val);
          if (s.ok()) {
            return true;
          }
          LOG(WARNING) << "Failed to toggle compaction to " << enable
                       << " for shard " << shard << ", attempt=" << attempt
                       << ", max_retries=" << max_retries << std::endl;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return false;
      });
      futures.push_back(std::move(f));
    }
    auto results = folly::coro::blockingWait(folly::collectAll(futures));
    for (auto& result : results) {
      if (result.hasValue()) {
        CHECK(result.value())
            << "Failed to toggle compaction to " << enable << std::endl;
      } else {
        CHECK(false) << "Failed to toggle compaction to " << enable
                     << " with exception " << result.exception().what()
                     << std::endl;
      }
    }
    auto_compaction_enabled_ = enable;
  }

  int64_t get_max_D() override {
    return max_D_;
  }

  // collect mem usage on all db shards, checkout rocks_db_mem_properties
  std::vector<int64_t> get_mem_usage() {
    int num_mem_component = rocks_db_mem_properties.size();
    std::vector<int64_t> mem_usages(num_mem_component);
    for (auto& db : dbs_) {
      for (int i = 0; i < num_mem_component; i++) {
        std::string property = rocks_db_mem_properties[i];
        std::string val;
        db->GetProperty(property, &val);
        if (val != "") {
          if (i != 0) {
            mem_usages[i] += folly::to<int64_t>(val);
          } else {
            mem_usages[i] = folly::to<int64_t>(val);
          }
        }
      }
    }
    return mem_usages;
  }

  std::vector<double> get_rocksdb_io_duration(
      const int64_t step,
      const int64_t interval) {
    std::vector<double> ret;
    ret.reserve(5);
    if (step > 0 && step % interval == 0) {
      int64_t reset_val = 0;
      auto read_dur = read_total_duration_.exchange(reset_val);

      auto fwd_rocksdb_read_dur = fwd_rocksdb_read_dur_.exchange(reset_val);
      auto fwd_l1_eviction_dur = fwd_l1_eviction_dur_.exchange(reset_val);
      auto bwd_l1_cnflct_miss_write_back_dur =
          bwd_l1_cnflct_miss_write_back_dur_.exchange(reset_val);
      auto flush_write_dur = flush_write_dur_.exchange(reset_val);

      ret.push_back(double(read_dur) / interval);
      ret.push_back(double(fwd_rocksdb_read_dur) / interval);
      ret.push_back(double(fwd_l1_eviction_dur) / interval);
      ret.push_back(double(bwd_l1_cnflct_miss_write_back_dur) / interval);
      ret.push_back(double(flush_write_dur) / interval);
    }
    return ret;
  }

  void compact() override {
    for (auto& db : dbs_) {
      db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
    }
  }

  void flush() {
    kv_db::EmbeddingKVDB::flush();
    for (auto& db : dbs_) {
      db->Flush(rocksdb::FlushOptions());
    }
  }

  int64_t num_shards() const {
    return dbs_.size();
  }

 private:
  void flush_or_compact(const int64_t timestep) override {
    // Only do manual Flush/Compactions if enabled
    if (memtable_flush_period_ > 0) {
      {
        RECORD_USER_SCOPE("FlushCompactIfNecessary");
        if (!done_staggered_flushes_) {
          flush_if_necessary(timestep);
        } else {
          compact_if_necessary(timestep);
        }
      }
    }
  }

  void flush_if_necessary(const int64_t timestep) {
    for (int64_t i = 0; i < dbs_.size(); i++) {
      if (shard_flush_compaction_deadlines_[i] == timestep) {
        rocksdb::FlushOptions fo;
        fo.wait = false;
        fo.allow_write_stall = false;
        dbs_[i]->Flush(fo);
        if (i == dbs_.size() - 1) {
          done_staggered_flushes_ = true;
          int64_t period_per_shard = compaction_period_ / dbs_.size();
          int64_t offset = memtable_flush_offset_ + compaction_period_;
          for (int64_t j = 0; j < dbs_.size(); j++) {
            shard_flush_compaction_deadlines_[j] =
                offset + (j * period_per_shard);
          }
        }
      }
    }
  }

  void compact_if_necessary(const int64_t timestep) {
    for (int64_t i = 0; i < dbs_.size(); i++) {
      if (shard_flush_compaction_deadlines_[i] == timestep) {
        rocksdb::ColumnFamilyMetaData meta;
        dbs_[i]->GetColumnFamilyMetaData(&meta);
        int32_t num_level0 = meta.levels[0].files.size();
        if (num_level0 >= l0_files_per_compact_) {
          dbs_[i]->CompactRange(
              rocksdb::CompactRangeOptions(), nullptr, nullptr);
        }
        shard_flush_compaction_deadlines_[i] += compaction_period_;
      }
    }
  }

  int64_t get_dim_from_index(int64_t index) const {
    if (sub_table_dims_.empty()) {
      return max_D_;
    }
    for (int i = 0; i < sub_table_hash_cumsum_.size(); i++) {
      if (index < sub_table_hash_cumsum_[i]) {
        return sub_table_dims_[i];
      }
    }
    return max_D_;
  }

  void fill_from_row_storage(
      int shard_id,
      unsigned char* weights_data_ptr,
      int64_t weights_row_index,
      unsigned char* row_storage_data_ptr,
      const rocksdb::Slice& weight_key) {
    // get the exact row dimension from the feature dimension list, so that
    // we can fill the correct number of elements into the weights tensor
    // for the given row. the rest of the row will stay as 0s, so untrained
    auto weight_index = *reinterpret_cast<const int64_t*>(weight_key.data());
    auto dim = get_dim_from_index(weight_index);
    int64_t row_width = elem_size_ * max_D_;
    auto copy_width = elem_size_ * dim;
    CHECK_LE(copy_width, row_width);
    int64_t row_index;
    initializers_[shard_id]->producer_queue_.dequeue(row_index);
    auto copied = std::copy_n(
        &(row_storage_data_ptr[row_index * row_width]),
        copy_width,
        &(weights_data_ptr[weights_row_index * row_width]));
    std::fill(
        copied, &(weights_data_ptr[(weights_row_index + 1) * row_width]), 0);
    initializers_[shard_id]->consumer_queue_.enqueue(row_index);
  }

  // use this iterator approach only when the keys in indices are contiguous
  // and matches the key order as defined by the comparator. i.e. the values
  // returned by the itertor are not thrown away. in other cases using an
  // iterator doesn't provide performance benefits.
  template <typename VALUE_T>
  void ssd_get_weights_iterator(
      const std::vector<rocksdb::Slice>& keys,
      const std::vector<int32_t>& key_indices,
      VALUE_T* weights_data_ptr,
      std::vector<rocksdb::ColumnFamilyHandle*>& cfs,
      int shard_id,
      rocksdb::ReadOptions local_ro,
      VALUE_T* row_storage_data_ptr) {
    auto iterator_ro = local_ro;
    iterator_ro.total_order_seek = true; // disable prefix filter
    auto it = dbs_[shard_id]->NewIterator(iterator_ro, cfs[0]);

    it->Seek(keys[0]);
    for (auto j = 0; j < keys.size(); ++j) {
      int64_t i = key_indices[j];
      if (!it->Valid()) {
        CHECK(it->status().ok());
        // the iterator reaches the end of the data,
        // generate a new row on the fly
        fill_from_row_storage(
            shard_id,
            reinterpret_cast<unsigned char*>(weights_data_ptr),
            i,
            reinterpret_cast<unsigned char*>(row_storage_data_ptr),
            keys[j]);
        continue;
      }

      const rocksdb::Slice& expected_key = keys[j];
      if (it->key().compare(expected_key) != 0) {
        // the row being looked up doesn't exist in
        // RocksDB, generate a new row on the fly
        fill_from_row_storage(
            shard_id,
            reinterpret_cast<unsigned char*>(weights_data_ptr),
            i,
            reinterpret_cast<unsigned char*>(row_storage_data_ptr),
            keys[j]);
      } else {
        const auto value = it->value();
        if (!std::is_same<VALUE_T, uint8_t>::value) {
          CHECK_EQ(value.size(), max_D_ * sizeof(VALUE_T));
        }
        std::copy(
            reinterpret_cast<const VALUE_T*>(value.data()),
            reinterpret_cast<const VALUE_T*>(value.data() + value.size()),
            &(weights_data_ptr[i * max_D_]));

        it->Next();
      }
    }

    delete it;
  }

  template <typename VALUE_T>
  void ssd_get_weights_multi_get(
      const std::vector<rocksdb::Slice>& keys,
      const std::vector<int32_t>& key_indices,
      VALUE_T* weights_data_ptr,
      std::vector<rocksdb::ColumnFamilyHandle*>& cfs,
      int shard_id,
      rocksdb::ReadOptions local_ro,
      VALUE_T* row_storage_data_ptr) {
    FOLLY_DECLARE_REUSED(values, std::vector<rocksdb::PinnableSlice>);
    FOLLY_DECLARE_REUSED(statuses, std::vector<rocksdb::Status>);
    values.resize(keys.size());
    statuses.resize(keys.size());
    dbs_[shard_id]->MultiGet(
        local_ro,
        keys.size(),
        cfs.data(),
        keys.data(),
        values.data(),
        statuses.data(),
        /*sorted_input=*/true);
    for (auto j = 0; j < keys.size(); ++j) {
      const auto& s = statuses[j];
      int64_t i = key_indices[j];
      const auto& value = values[j];
      if (s.ok()) {
        if (!std::is_same<VALUE_T, uint8_t>::value) {
          CHECK_EQ(value.size(), max_D_ * sizeof(VALUE_T));
        }
        std::copy(
            reinterpret_cast<const VALUE_T*>(value.data()),
            reinterpret_cast<const VALUE_T*>(value.data() + value.size()),
            &(weights_data_ptr[i * max_D_]));
      } else {
        CHECK(s.IsNotFound());
        fill_from_row_storage(
            shard_id,
            reinterpret_cast<unsigned char*>(weights_data_ptr),
            i,
            reinterpret_cast<unsigned char*>(row_storage_data_ptr),
            keys[j]);
      }
    }
  }

  // use_iterator=true is only efficient when the key sequence being looked up
  // matches the key sequence matches the key sequence obtained through the
  // iterator. see the comment for ssd_get_weights_iterator.
  template <bool use_iterator>
  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async_impl(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const SnapshotHandle* snapshot_handle) {
    RECORD_USER_SCOPE("EmbeddingRocksDB::get");
#ifdef FBGEMM_FBCODE
    auto start_ts = facebook::WallClockUtil::NowInUsecFast();
#endif
    std::vector<folly::Future<folly::Unit>> futures;
    auto count_ = count.scalar_type() == at::ScalarType::Long
        ? *(count.data_ptr<int64_t>())
        : *(count.data_ptr<int32_t>());

    for (auto shard = 0; shard < dbs_.size(); ++shard) {
      // Get a snapshot for the shard
      snapshot_ptr_t snapshot = snapshot_handle == nullptr
          ? nullptr
          : snapshot_handle->get_snapshot_for_shard(shard);
      auto local_ro = ro_;
      local_ro.snapshot = snapshot;
      auto f =
          folly::via(executor_.get())
              .thenValue([=, &indices, &weights](folly::Unit) {
                FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
                    weights.scalar_type(), "ssd_get", [&] {
                      using value_t = scalar_t;
                      FBGEMM_DISPATCH_INTEGRAL_TYPES(
                          indices.scalar_type(), "ssd_get", [&] {
                            using index_t = scalar_t;
                            CHECK(indices.is_contiguous());
                            CHECK(weights.is_contiguous());
                            auto indices_data_ptr = indices.data_ptr<index_t>();
                            auto D = weights.size(1);
                            CHECK_EQ(indices.size(0), weights.size(0));
                            CHECK_EQ(D, max_D_);
                            auto weights_data_ptr = weights.data_ptr<value_t>();
                            FOLLY_DECLARE_REUSED(
                                keys, std::vector<rocksdb::Slice>);
                            FOLLY_DECLARE_REUSED(
                                key_indices, std::vector<int32_t>);
                            FOLLY_DECLARE_REUSED(
                                cfs, std::vector<rocksdb::ColumnFamilyHandle*>);
                            FOLLY_DECLARE_REUSED(
                                values, std::vector<rocksdb::PinnableSlice>);
                            FOLLY_DECLARE_REUSED(
                                statuses, std::vector<rocksdb::Status>);
                            auto* dcf = dbs_[shard]->DefaultColumnFamily();
                            for (auto i = 0; i < count_; ++i) {
                              // "no-op"/empty evicted tensor
                              if (indices_data_ptr[i] == -1) {
                                continue;
                              }
                              if (kv_db_utils::hash_shard(
                                      indices_data_ptr[i], dbs_.size()) !=
                                  shard) {
                                continue;
                              }
                              key_indices.push_back(i);
                            }

                            // bail if nothing to do
                            if (key_indices.empty()) {
                              return;
                            }

                            std::sort(
                                key_indices.begin(),
                                key_indices.end(),
                                [&](int32_t lhs, int32_t rhs) {
                                  auto lhs_key = indices_data_ptr[lhs];
                                  auto rhs_key = indices_data_ptr[rhs];
                                  return lhs_key < rhs_key;
                                });
                            for (const auto& i : key_indices) {
                              const auto key = rocksdb::Slice(
                                  reinterpret_cast<const char*>(
                                      &(indices_data_ptr[i])),
                                  sizeof(index_t));
                              keys.push_back(key);
                              cfs.push_back(dcf);
                            }
                            CHECK_EQ(key_indices.size(), keys.size());
                            CHECK_EQ(key_indices.size(), cfs.size());

                            const auto& init_storage =
                                initializers_[shard]->row_storage_;
                            // Sanity check
                            TORCH_CHECK(
                                init_storage.scalar_type() ==
                                    weights.scalar_type(),
                                "init_storage (",
                                toString(init_storage.scalar_type()),
                                ") and weights scalar (",
                                toString(weights.scalar_type()),
                                ") types mismatch");
                            auto row_storage_data_ptr =
                                init_storage.data_ptr<value_t>();
                            if (use_iterator) {
                              ssd_get_weights_iterator(
                                  keys,
                                  key_indices,
                                  weights_data_ptr,
                                  cfs,
                                  shard,
                                  local_ro,
                                  row_storage_data_ptr);
                            } else {
                              ssd_get_weights_multi_get(
                                  keys,
                                  key_indices,
                                  weights_data_ptr,
                                  cfs,
                                  shard,
                                  local_ro,
                                  row_storage_data_ptr);
                            }
                          });
                    });
              });
      futures.push_back(std::move(f));
    }
#ifdef FBGEMM_FBCODE
    auto duration = facebook::WallClockUtil::NowInUsecFast() - start_ts;
    read_total_duration_ += duration;
#endif
    return folly::collect(futures);
  }

  friend class SnapshotHandle;

  std::vector<std::unique_ptr<rocksdb::DB>> dbs_;
  std::vector<std::unique_ptr<Initializer>> initializers_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  rocksdb::ReadOptions ro_{};
  rocksdb::WriteOptions wo_{};
  std::shared_ptr<rocksdb::RateLimiter> rate_limiter_;
  std::vector<int64_t> shard_flush_compaction_deadlines_;
  bool done_staggered_flushes_;
  int64_t memtable_flush_offset_;
  int64_t memtable_flush_period_;
  int64_t compaction_period_;
  int64_t l0_files_per_compact_;
  bool auto_compaction_enabled_;

  // break down on rocksdb write duration for details checkout
  // RocksdbWriteMode
  std::atomic<int64_t> read_total_duration_{0};
  std::atomic<int64_t> fwd_rocksdb_read_dur_{0};
  std::atomic<int64_t> fwd_l1_eviction_dur_{0};
  std::atomic<int64_t> bwd_l1_cnflct_miss_write_back_dur_{0};
  std::atomic<int64_t> flush_write_dur_{0};

  std::unordered_map<const SnapshotHandle*, std::unique_ptr<SnapshotHandle>>
      snapshots_;
  int64_t max_D_;
  int64_t elem_size_;
  std::vector<int64_t> sub_table_dims_;
  std::vector<int64_t> sub_table_hash_cumsum_;
}; // class EmbeddingRocksDB

} // namespace ssd
