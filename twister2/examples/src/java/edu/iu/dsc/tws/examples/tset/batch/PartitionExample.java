//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

package edu.iu.dsc.tws.examples.tset.batch;

import java.util.HashMap;
import java.util.Iterator;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.api.tset.fn.ComputeCollectorFunc;
import edu.iu.dsc.tws.api.tset.fn.ComputeFunc;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.fn.LoadBalancePartitioner;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

public class PartitionExample extends BatchTsetExample {
  private static final Logger LOG = Logger.getLogger(PartitionExample.class.getName());
  private static final long serialVersionUID = -2753072757838198105L;

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);
    int start = env.getWorkerID() * 100;
    SourceTSet<Integer> src = dummySource(env, start, COUNT, PARALLELISM);

    LOG.info("test foreach");
    src.partition(new LoadBalancePartitioner<>())
        .forEach(i -> LOG.info("foreach: " + i));

    LOG.info("test map");
    src.partition(new LoadBalancePartitioner<>())
        .map(i -> i.toString() + "$$")
        .direct()
        .forEach(s -> LOG.info("map: " + s));

    LOG.info("test flat map");
    src.partition(new LoadBalancePartitioner<>())
        .flatmap((i, c) -> c.collect(i.toString() + "##"))
        .direct()
        .forEach(s -> LOG.info("flat:" + s));

    LOG.info("test compute");
    src.partition(new LoadBalancePartitioner<>())
        .compute((ComputeFunc<Iterator<Integer>, Integer>) input -> {
          int sum = 0;
          while (input.hasNext()) {
            sum += input.next();
          }
          return sum;
        })
        .direct()
        .forEach(i -> LOG.info("comp: " + i));

    LOG.info("test computec");
    src.partition(new LoadBalancePartitioner<>())
        .compute((ComputeCollectorFunc<Iterator<Integer>, String>)
            (input, output) -> {
              int sum = 0;
              while (input.hasNext()) {
                sum += input.next();
              }
              output.collect("sum" + sum);
            })
        .direct()
        .forEach(s -> LOG.info("computec: " + s));
  }


  public static void main(String[] args) {
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    JobConfig jobConfig = new JobConfig();
    BatchTsetExample.submitJob(config, PARALLELISM, jobConfig, PartitionExample.class.getName());
  }
}
