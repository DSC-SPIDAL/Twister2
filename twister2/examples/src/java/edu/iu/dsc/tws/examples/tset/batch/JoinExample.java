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
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.comms.CommunicationContext;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.links.batch.JoinTLink;
import edu.iu.dsc.tws.tset.sets.batch.KeyedTSet;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

// todo check the correctness of INNER joins!
public class JoinExample extends BatchTsetExample {
  private static final Logger LOG = Logger.getLogger(JoinExample.class.getName());

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);
    int para = 2;
    int workerID = env.getWorkerID();
    SourceTSet<Integer> src0 = dummySource(env, COUNT, para).setName("src0");

    KeyedTSet<Integer, Integer> left = src0.mapToTuple(i -> new Tuple<>(i % 2, i)).setName("left");
    left.keyedDirect().forEach(i -> LOG.info(workerID + "L " + i.toString()));

    SourceTSet<Integer> src1 = dummySource(env, COUNT, para).setName("src1");
    KeyedTSet<Integer, Integer> right = src1.mapToTuple(i -> new Tuple<>(i % 2, i)).setName(
        "right");
    right.keyedDirect().forEach(i -> LOG.info(workerID + "R " + i.toString()));

    JoinTLink<Integer, Integer, Integer> join =
        left.join(right, CommunicationContext.JoinType.INNER, Integer::compareTo)
            .setName("join");

    join.forEach(t -> LOG.info(workerID + "out: " + t.toString()));
  }

  public static void main(String[] args) {
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    JobConfig jobConfig = new JobConfig();
    BatchTsetExample.submitJob(config, PARALLELISM, jobConfig, JoinExample.class.getName());
  }
}
