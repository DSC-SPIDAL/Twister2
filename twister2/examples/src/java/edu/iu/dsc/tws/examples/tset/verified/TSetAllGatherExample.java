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
import java.util.Iterator;
import java.util.List;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.compute.OperationNames;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.api.tset.TSetContext;
import edu.iu.dsc.tws.api.tset.fn.SinkFunc;
import edu.iu.dsc.tws.examples.tset.BaseTSetBatchWorker;
import edu.iu.dsc.tws.examples.verification.VerificationException;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.links.batch.AllGatherTLink;
import edu.iu.dsc.tws.tset.sets.batch.SinkTSet;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

public class TSetAllGatherExample extends BaseTSetBatchWorker {
  private static final Logger LOG = Logger.getLogger(TSetAllGatherExample.class.getName());

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    super.execute(workerEnv);
    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);

    // set the parallelism of source to task stage 0
    List<Integer> taskStages = jobParameters.getTaskStages();
    int sourceParallelism = taskStages.get(0);
    int sinkParallelism = taskStages.get(1);
    SourceTSet<int[]> source =
        env.createSource(new TestBaseSource(), sourceParallelism).setName("Source");

    AllGatherTLink<int[]> gather = source.allGather();

    SinkTSet<Iterator<Tuple<Integer, int[]>>> sink =
        gather.sink(new SinkFunc<Iterator<Tuple<Integer, int[]>>>() {
          private TSetContext context;

          @Override
          public boolean add(Iterator<Tuple<Integer, int[]>> value) {
            // todo: check this!
            int[] result = new int[0];
            while (value.hasNext()) {
              Tuple<Integer, int[]> t = value.next();
              if (t.getKey().equals(context.getIndex())) {
                result = t.getValue();
                break;
              }
            }

            LOG.info("Task Id : " + context.getIndex()
                + " Results " + Arrays.toString(result));
            experimentData.setOutput(value);
            try {
              verify(OperationNames.ALLGATHER);
            } catch (VerificationException e) {
              LOG.info("Exception Message : " + e.getMessage());
            }
            return true;
          }

          @Override
          public void prepare(TSetContext ctx) {
            this.context = ctx;
          }
        });
    env.run(sink);
  }

}
