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

package edu.iu.dsc.tws.examples.tset.verified;

import java.util.Arrays;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.compute.OperationNames;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.examples.tset.BaseTSetBatchWorker;
import edu.iu.dsc.tws.examples.verification.VerificationException;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.links.batch.AllReduceTLink;
import edu.iu.dsc.tws.tset.sets.batch.SinkTSet;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

public class TSetAllReduceExample extends BaseTSetBatchWorker {
  private static final Logger LOG = Logger.getLogger(TSetAllReduceExample.class.getName());

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    super.execute(workerEnv);
    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);

    // set the parallelism of source to task stage 0
    int srcPara = jobParameters.getTaskStages().get(0);
    int sinkPara = jobParameters.getTaskStages().get(1);
    SourceTSet<int[]> source = env.createSource(new TestBaseSource(), srcPara)
        .setName("Source");
    AllReduceTLink<int[]> reduce = source.allReduce((t1, t2) -> {
      int[] val = new int[t1.length];
      for (int i = 0; i < t1.length; i++) {
        val[i] = t1[i] + t2[i];
      }
      return val;
    });

    SinkTSet<int[]> sink = reduce.sink(value -> {
      experimentData.setOutput(value);
      try {
        LOG.info("Results " + Arrays.toString(value));
        verify(OperationNames.ALLREDUCE);
      } catch (VerificationException e) {
        LOG.info("Exception Message : " + e.getMessage());
      }
      return true;
    });
    env.run(sink);

  }

}
