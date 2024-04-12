import { Component, OnInit } from '@angular/core';
import { LoadingService } from '../../../Services/loading/loading.service';

@Component({
  selector: 'app-loading',
  templateUrl: './loading.component.html',
  styleUrl: './loading.component.css'
})
export class LoadingComponent implements OnInit{
  isloading!:boolean;
  constructor(logingservices:LoadingService){
    logingservices.isLoading.subscribe((loading)=>{
      this.isloading=loading;
    })
    // logingservices.showLoading()
  }
  ngOnInit(): void {

  }
}
