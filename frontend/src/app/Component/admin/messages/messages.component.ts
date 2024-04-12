import { Component,OnInit } from '@angular/core';
import { Contect } from '../../../model/contect';
import { AdminService } from '../../../Services/admin/admin.service';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-messages',
  templateUrl: './messages.component.html',
  styleUrl: './messages.component.css'
})
export class MessagesComponent implements OnInit{
  message?:Contect[];
  responce:string="";
  constructor(private adminservices:AdminService,private toaster:ToastrService){}
  ngOnInit(): void {
    this.adminservices.getAllmessage().subscribe((message)=>{
      this.message=message;
      console.log(this.message)
    })

  }
  send(id:string){

    const x:Contect={
      id: '',
      name: '',
      email: '',
      message: '',
      date: '',
      response: this.responce
    }
    this.adminservices.sendMessage(id,x).subscribe((m)=>{
      this.toaster.success("Successfully sended");
      this.adminservices.getAllmessage().subscribe((message)=>{
        this.message=message;

      })
    })
  }
}
