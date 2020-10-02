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

package edu.iu.dsc.tws.api.compute.nodes;

import edu.iu.dsc.tws.api.compute.TaskContext;
import edu.iu.dsc.tws.api.compute.modifiers.Closable;
import edu.iu.dsc.tws.api.config.Config;

public abstract class BaseNode implements INode, Closable {

  protected TaskContext context;

  protected Config config;

  @Override
  public void prepare(Config cfg, TaskContext ctx) {
    this.config = cfg;
    this.context = ctx;
  }

  @Override
  public void close() {

  }

  @Override
  public void reset() {

  }
}
