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
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.api.tset.fn.ComputeCollectorFunc;
import edu.iu.dsc.tws.api.tset.fn.ComputeFunc;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.links.batch.KeyedReduceTLink;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

public class KReduceExample extends BatchTsetExample {
  private static final Logger LOG = Logger.getLogger(KReduceExample.class.getName());
  private static final long serialVersionUID = -2753072757838198105L;

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);
    int start = env.getWorkerID() * 100;
    SourceTSet<Integer> src = dummySource(env, start, COUNT, PARALLELISM);

    KeyedReduceTLink<Integer, Integer> kreduce = src.mapToTuple(i -> new Tuple<>(i % 10, i))
        .keyedReduce(Integer::sum);

    LOG.info("test foreach");
    kreduce.forEach(t -> LOG.info("sum by key=" + t.getKey() + ", " + t.getValue()));

    LOG.info("test map");
    kreduce.map(i -> i.toString() + "$$")
        .direct()
        .forEach(s -> LOG.info("map: " + s));

    LOG.info("test compute");
    kreduce.compute(
        (ComputeFunc<String, Iterator<Tuple<Integer, Integer>>>) input -> {
          StringBuilder s = new StringBuilder();
          while (input.hasNext()) {
            s.append(input.next().toString()).append(" ");
          }
          return s.toString();
        })
        .direct()
        .forEach(s -> LOG.info("compute: concat " + s));

    LOG.info("test computec");
    kreduce.compute((ComputeCollectorFunc<String, Iterator<Tuple<Integer, Integer>>>)
        (input, output) -> {
          while (input.hasNext()) {
            output.collect(input.next().toString());
          }
        })
        .direct()
        .forEach(s -> LOG.info("computec: " + s));
  }


  public static void main(String[] args) {
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    JobConfig jobConfig = new JobConfig();
    BatchTsetExample.submitJob(config, PARALLELISM, jobConfig, KReduceExample.class.getName());
  }
}
