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
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.links.batch.DirectTLink;
import edu.iu.dsc.tws.tset.sets.batch.ComputeTSet;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

/*
 * Can be removed!
 */
public class EvaluateExample extends BatchTsetExample {
  private static final Logger LOG = Logger.getLogger(EvaluateExample.class.getName());
  private static final long serialVersionUID = -2753072757838198105L;

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);
    SourceTSet<Integer> src = dummyReplayableSource(env, COUNT, PARALLELISM).setName("src");

    DirectTLink<Integer> direct = src.direct().setName("direct");

    LOG.info("test foreach");
    ComputeTSet<Object, Iterator<Integer>> tset1 =
        direct.lazyForEach(i -> LOG.info("foreach: " + i));

    LOG.info("test map");
    ComputeTSet<Object, Iterator<String>> tset2 =
        direct.map(i -> i.toString() + "$$").setName("map")
            .direct()
            .lazyForEach(s -> LOG.info("map: " + s));

    for (int i = 0; i < 4; i++) {
      LOG.info("iter " + i);
      env.eval(tset1);
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
//      env.eval(tset2);
    }

    env.finishEval(tset1);
//    env.finishEval(tset2);

/*    LOG.info("test flat map");
    direct.flatmap((i, c) -> c.collect(i.toString() + "##")).setName("flatmap")
        .direct()
        .lazyForEach(s -> LOG.info("flat:" + s));

    LOG.info("test compute");
    direct.compute((ComputeFunc<String, Iterator<Integer>>) input -> {
      int sum = 0;
      while (input.hasNext()) {
        sum += input.next();
      }
      return "sum" + sum;
    }).setName("compute")
        .direct()
        .lazyForEach(i -> LOG.info("comp: " + i));

    LOG.info("test computec");
    direct.compute((ComputeCollectorFunc<String, Iterator<Integer>>)
        (input, output) -> {
          int sum = 0;
          while (input.hasNext()) {
            sum += input.next();
          }
          output.collect("sum" + sum);
        }).setName("ccompute")
        .direct()
        .lazyForEach(s -> LOG.info("computec: " + s));

    LOG.info("test sink");
    direct.sink((SinkFunc<Iterator<Integer>>) value -> {
      while (value.hasNext()) {
        LOG.info("val =" + value.next());
      }
      return true;
    });*/

  }


  public static void main(String[] args) {
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    JobConfig jobConfig = new JobConfig();
    BatchTsetExample.submitJob(config, PARALLELISM, jobConfig, EvaluateExample.class.getName());
  }
}
